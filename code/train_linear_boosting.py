"""Frozen candidate grids; training only fit, validation selection, held-out evaluation."""
import argparse, json, time, warnings
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from data_protocol import ROOT,read_json,write_json
from release_common import sha,load_cache
from experiment import PrefixDataset,baseline_features,environment,save_predictions,open_run
from evaluation import probability_metrics

def candidates(name,seed):
    if name in ('gold_logistic','full_feature_lr'):
        if name=='full_feature_lr':
            params=read_json(ROOT.parent/'baseline_grid.json')['full_feature_lr']['fixed']
            params['random_state']=seed
            return [make_pipeline(StandardScaler(),LogisticRegression(C=c,**params)) for c in (.1,1.,10.)]
        return [make_pipeline(StandardScaler(),LogisticRegression(C=c,random_state=seed,
                    max_iter=1000 if name=='gold_logistic' else 5000)) for c in (.1,1.,10.)]
    from lightgbm import LGBMClassifier
    grid=read_json(ROOT.parent/'baseline_grid.json')['lightgbm']['fixed']
    grid['random_state']=seed
    return [LGBMClassifier(num_leaves=n,**grid) for n in (15,31,63)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=['gold_logistic','full_feature_lr','lightgbm'],default=['gold_logistic','full_feature_lr','lightgbm'])
    p.add_argument('--minutes',nargs='+',type=int,choices=[10,20,30,40],default=[10,20,30,40])
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);args=p.parse_args()
    meta=load_cache(args.cache)
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for minute in args.minutes:
        data={s:PrefixDataset(args.cache/f'{s}_{minute}m.npz') for s in ('train','validation','test')}
        for name in args.models:
            x={s:baseline_features(d,simple=name=='gold_logistic') for s,d in data.items()}
            for seed in args.seeds:
                folder=args.output/f'{name}_{minute}m_seed{seed}'
                spec={'model':name,'minute':minute,'seed':seed,'cache_id':meta['cache_id'],
                      'cache_files':meta['files'],'environment':environment(),
                      'fit_predict_threadpool_limit':None if name=='gold_logistic' else 4,
                      'source_hashes':{f.name:sha(f) for f in ROOT.glob('*.py')},'grid_sha256':sha(ROOT.parent/'baseline_grid.json')}
                if open_run(folder,spec,resume=True) is not None:continue
                start=time.perf_counter();best=None;best_loss=float('inf');search=[]
                for index,model in enumerate(candidates(name,seed)):
                    with (nullcontext() if name=='gold_logistic' else threadpool_limits(limits=4)):
                        model.fit(x['train'],data['train'].arrays['outcome'])
                        probs=model.predict_proba(x['validation'])[:,1]
                    metrics=probability_metrics(data['validation'].arrays['outcome'],probs)
                    c=folder/f'candidate_{index}';c.mkdir(exist_ok=True)
                    joblib.dump(model,c/'model.joblib')
                    save_predictions(c/'validation_predictions.npz',data['validation'],data['validation'].arrays['outcome'],probs,data['validation'].arrays['match_id'])
                    search.append({'candidate':index,'validation':metrics,'parameters':str(model)})
                    if metrics['log_loss']<best_loss:best=model;best_loss=metrics['log_loss'];chosen=index
                joblib.dump(best,folder/'model.joblib')
                result={'model':name,'minute':minute,'seed':seed,'best_candidate':chosen,'search':search,'specification':spec}
                for split in ('validation','test'):
                    with (nullcontext() if name=='gold_logistic' else threadpool_limits(limits=4)):
                        probs=best.predict_proba(x[split])[:,1]
                    save_predictions(folder/('predictions.npz' if split=='test' else 'validation_predictions.npz'),data[split],data[split].arrays['outcome'],probs,data[split].arrays['match_id'])
                    result[split]=probability_metrics(data[split].arrays['outcome'],probs)
                result['runtime_seconds']=time.perf_counter()-start
                write_json(folder/'result.json',result)
                print('COMPLETE',folder.name,flush=True)

if __name__=='__main__':main()
