"""Corrected-input replication of the paper's neural architecture comparisons."""
import argparse
import gc
import json
import time

import numpy as np
import torch

from data_protocol import ROOT, read_json, write_json, fingerprint
from evaluation import probability_metrics
from experiment import PrefixDataset, atomic_torch_save, collate_prefixes, environment, seed_everything, to_device, open_run, validate_progress
from feature_extractor import feature_contract
from models import DotaMultiModalPredictor, VARIANTS
from release_common import sha
from release_common import load_cache
from pathlib import Path
CACHE=None

RESULTS=None
VARIANT_ORDER=('full','no_graph','gru','no_role','complete_graph','no_draft_branch',
               'mean_draft','single_task','temporal_only')
MINUTES=(10,20,30,40)
SEEDS=(17,29,43)
SOURCES=tuple(p.name for p in ROOT.glob('*.py'))


def on_device(dataset,device):
    return {k:torch.from_numpy(v).to(device) for k,v in dataset.arrays.items()}


def select(arrays,indices,length):
    selected={k:v[indices] for k,v in arrays.items()}
    selected['lengths']=torch.full((len(indices),),length,dtype=torch.int64,device=indices.device)
    return selected


def evaluate(model,arrays,device,length):
    model.eval();probs=[]
    with torch.inference_mode():
        for start in range(0,len(arrays['outcome']),64):
            idx=torch.arange(start,min(start+64,len(arrays['outcome'])),device=device)
            probs.append(model(select(arrays,idx,length))['logits'].sigmoid().cpu().numpy())
    return np.concatenate(probs)


def save_predictions(path,dataset,probabilities):
    a=dataset.arrays
    np.savez_compressed(path,match_id=a['match_id'],labels=a['outcome'],probabilities=probabilities,
                        patch=a['patch'],start_time=a['start_time'],gold_difference=a['sequence'][:,-1,0])


def train(cache_meta,minute,variant,seed):
    torch.set_num_threads(4);seed_everything(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder=RESULTS/f'{variant}_{minute}m_seed{seed}'
    spec=json.loads(json.dumps({'minute':minute,'model':variant,'seed':seed,
        'cache_id':cache_meta['cache_id'],'cache_files':cache_meta['files'],
        'source_hashes':{name:sha(ROOT/name) for name in SOURCES},
        'environment':environment(),'epochs':30,'batch_size':64,'patience':5,
        'optimizer':'Adam','learning_rate':.001,'weight_decay':1e-4,
        'aux_loss_weights':[0,0] if variant=='single_task' else [.1,.1],
        'gradient_clip':5,'selection':'validation_log_loss',
        'training_implementation':'on_device_batch_indexing'}))
    completed=open_run(folder,spec,resume=True)
    if completed is not None:
        print('Already complete',folder.name,flush=True)
        return completed
    datasets={s:PrefixDataset(CACHE/f'{s}_{minute}m.npz') for s in ('train','validation','test')}
    arrays={s:on_device(d,device) for s,d in datasets.items()}
    length=minute+1
    if datasets['train'].arrays['sequence'].shape[1]!=length:raise ValueError('Wrong sequence length')
    original=to_device(collate_prefixes([datasets['train'][i] for i in (0,1,2)]),device)
    direct=select(arrays['train'],torch.tensor([0,1,2],device=device),length)
    for key in original:
        if not torch.equal(original[key],direct[key]):raise ValueError('Fast batch mismatch: '+key)
    generator=torch.Generator().manual_seed(seed)
    model=DotaMultiModalPredictor(**VARIANTS[variant]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    best_loss,best_state,best_epoch,stale,history=float('inf'),None,0,0,[]
    first_epoch,elapsed=1,0.
    if (folder/'progress.pt').exists():
        p=torch.load(folder/'progress.pt',map_location='cpu',weights_only=True)
        validate_progress(p,spec)
        model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer'])
        best_loss,best_state,best_epoch=p['best_loss'],p['best_state'],p['best_epoch']
        stale,history,elapsed=p['stale'],p['history'],p['elapsed']
        first_epoch=p['epoch']+1
        torch.set_rng_state(p['torch_rng']);generator.set_state(p['loader_rng'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(p['cuda_rng'])
        print('Resuming',folder.name,'at epoch',first_epoch,flush=True)
    start=time.perf_counter();n=len(datasets['train'])
    for epoch in range(first_epoch,31):
        if stale>=5:break
        model.train();totals=np.zeros(3)
        order=torch.randperm(n,generator=generator)
        for first in range(0,n,64):
            indices=order[first:first+64].to(device)
            batch=select(arrays['train'],indices,length)
            optimizer.zero_grad(set_to_none=True)
            out=model(batch)
            win_loss=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],batch['outcome'])
            squared=(out['aux']-batch['aux']).square()
            aux=(squared*batch['aux_mask'][:,None]).sum(0)/batch['aux_mask'].sum().clamp_min(1)
            loss=win_loss if variant=='single_task' else win_loss+.1*aux.sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            totals+=np.array([win_loss.item(),*aux.detach().cpu().tolist()])*len(indices)
        probabilities=evaluate(model,arrays['validation'],device,length)
        metrics=probability_metrics(datasets['validation'].arrays['outcome'],probabilities)
        if not np.isfinite(metrics['log_loss']):raise ValueError('Invalid validation loss')
        history.append({'epoch':epoch,'train_loss_components':(totals/n).tolist(),'validation':metrics})
        if metrics['log_loss']<best_loss:
            best_loss,best_epoch,stale=metrics['log_loss'],epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        atomic_torch_save({'specification_sha256':fingerprint(spec),'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'best_loss':best_loss,'best_state':best_state,'best_epoch':best_epoch,'stale':stale,
            'history':history,'elapsed':elapsed+time.perf_counter()-start,
            'torch_rng':torch.get_rng_state(),'loader_rng':generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},folder/'progress.pt')
        write_json(folder/'training_status.json',{'last_completed_epoch':epoch,'best_epoch':best_epoch,
            'best_validation_log_loss':best_loss,'test_evaluated':False})
        print(folder.name,'epoch',epoch,'val_ll',round(metrics['log_loss'],5),flush=True)
    if best_state is None:raise ValueError('Missing best state')
    model.load_state_dict(best_state)
    val_probs=evaluate(model,arrays['validation'],device,length)
    test_probs=evaluate(model,arrays['test'],device,length)
    save_predictions(folder/'validation_predictions.npz',datasets['validation'],val_probs)
    save_predictions(folder/'predictions.npz',datasets['test'],test_probs)
    atomic_torch_save({'state_dict':best_state,'model_config':model.config,'specification':spec,
        'feature_contract':feature_contract(),'tower_correction':'v4_objectives.py',
        'purchase_prior':read_json(CACHE/'purchase_prior.json'),'aux_horizon':5,
        'best_epoch':best_epoch},folder/'model.pt')
    result={'minute':minute,'model':variant,'seed':seed,
        'test':probability_metrics(datasets['test'].arrays['outcome'],test_probs),
        'validation':probability_metrics(datasets['validation'].arrays['outcome'],val_probs),
        'best_epoch':best_epoch,'best_validation_log_loss':best_loss,
        'parameters':sum(p.numel() for p in model.parameters()),
        'runtime_seconds':elapsed+time.perf_counter()-start,'history':history,'specification':spec}
    write_json(folder/'result.json',result)
    write_json(folder/'training_status.json',{'last_completed_epoch':history[-1]['epoch'],
        'best_epoch':best_epoch,'best_validation_log_loss':best_loss,'test_evaluated':True})
    print('COMPLETE',folder.name,'accuracy',round(result['test']['accuracy'],5),flush=True)
    del model,optimizer,arrays,datasets
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def main():
    global CACHE, RESULTS
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=VARIANT_ORDER,default=VARIANT_ORDER)
    parser.add_argument('--minutes',nargs='+',type=int,choices=MINUTES,default=MINUTES)
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=SEEDS)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    CACHE=args.cache.resolve();RESULTS=args.output.resolve()
    meta=load_cache(CACHE)
    for minute in args.minutes:
        for variant in args.models:
            for seed in args.seeds:train(meta,minute,variant,seed)


if __name__=='__main__':main()
