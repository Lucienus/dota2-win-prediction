"""Retrain the paper's tower-dependent conventional baselines on corrected inputs."""
import argparse
import json
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_protocol import ROOT, read_json, write_json
from evaluation import probability_metrics
from experiment import PrefixDataset, baseline_features, environment, save_predictions, open_run
from release_common import sha
from release_common import load_cache
from pathlib import Path
CACHE=None

RESULTS=None
MODELS=('random_forest','xgboost')
MINUTES=(10,20,30,40)
SEEDS=(17,29,43)
SOURCES=tuple(p.name for p in ROOT.glob('*.py'))


def train(meta,minute,name,seed):
    folder=RESULTS/f'{name}_{minute}m_seed{seed}'
    spec=json.loads(json.dumps({'minute':minute,'model':name,'seed':seed,
        'cache_id':meta['cache_id'],'cache_files':meta['files'],
        'source_hashes':{n:sha(ROOT/n) for n in SOURCES},'environment':environment(),
        'selection':'validation_log_loss','estimator_n_jobs':4,
        'feature_builder':'original_hero_onehot_512_and_corrected_nodes'}))
    completed=open_run(folder,spec,resume=True)
    if completed is not None:
        print('Already complete',folder.name,flush=True)
        return completed
    data={s:PrefixDataset(CACHE/f'{s}_{minute}m.npz') for s in ('train','validation','test')}
    x={s:baseline_features(d) for s,d in data.items()}
    y={s:d.arrays['outcome'] for s,d in data.items()}
    if name=='random_forest':
        candidates=[RandomForestClassifier(n_estimators=200,min_samples_leaf=leaf,
                     random_state=seed,n_jobs=4) for leaf in (1,5)]
    elif name=='xgboost':
        candidates=[XGBClassifier(n_estimators=200,max_depth=depth,learning_rate=.05,
                     subsample=.9,colsample_bytree=.9,eval_metric='logloss',random_state=seed,
                     n_jobs=4) for depth in (3,6)]
    else:raise ValueError('Unknown model')
    best,best_loss,search=None,float('inf'),[]
    start=time.perf_counter()
    for index,candidate in enumerate(candidates):
        candidate.fit(x['train'],y['train'])
        p=candidate.predict_proba(x['validation'])[:,1]
        metrics=probability_metrics(y['validation'],p)
        search.append({'candidate':index,'parameters':str(candidate),'validation':metrics})
        print(folder.name,'candidate',index,'validation_ll',round(metrics['log_loss'],5),flush=True)
        if metrics['log_loss']<best_loss:best,best_loss=candidate,metrics['log_loss']
    validation=best.predict_proba(x['validation'])[:,1]
    test=best.predict_proba(x['test'])[:,1]
    save_predictions(folder/'validation_predictions.npz',data['validation'],y['validation'],validation,
                     data['validation'].arrays['match_id'])
    save_predictions(folder/'predictions.npz',data['test'],y['test'],test,
                     data['test'].arrays['match_id'])
    joblib.dump(best,folder/'model.joblib')
    result={'minute':minute,'model':name,'seed':seed,'test':probability_metrics(y['test'],test),
            'validation':probability_metrics(y['validation'],validation),
            'best_candidate':search.index(next(s for s in search if s['validation']['log_loss']==best_loss)),
            'selection_metric':'validation_log_loss','search':search,
            'runtime_seconds':time.perf_counter()-start,'specification':spec}
    write_json(folder/'result.json',result)
    print('COMPLETE',folder.name,'accuracy',round(result['test']['accuracy'],5),flush=True)
    return result


def main():
    global CACHE, RESULTS
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--minutes',nargs='+',type=int,choices=MINUTES,default=MINUTES)
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=SEEDS)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    CACHE=args.cache.resolve();RESULTS=args.output.resolve();meta=load_cache(CACHE)
    for minute in args.minutes:
        for name in args.models:
            for seed in args.seeds:train(meta,minute,name,seed)


if __name__=='__main__':main()
