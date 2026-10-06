"""Fixed manuscript resampling rules; no seed-level pseudoreplication."""
from datetime import datetime, timezone
import json
import numpy as np
from evaluation import paired_bootstrap

def loss(y,p,cast=True):
    p=np.asarray(p,dtype=float) if cast else p
    p=np.clip(p,1e-7,1-1e-7)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))

def interval(d,groups=None,repeats=10000,seed=731):
    d=np.asarray(d,dtype=float)
    if not len(d) or not np.isfinite(d).all():raise ValueError('Invalid differences')
    rng=np.random.default_rng(seed);matches=np.empty(repeats);clusters=np.empty(repeats)
    if groups is not None:
        if len(groups)!=len(d):raise ValueError('Cluster length mismatch')
        unique,inverse=np.unique(groups,return_inverse=True)
        totals=np.bincount(inverse,weights=d);sizes=np.bincount(inverse)
    for i in range(repeats):
        matches[i]=d[rng.integers(len(d),size=len(d))].mean()
        if groups is not None:
            draw=rng.integers(len(unique),size=len(unique))
            clusters[i]=totals[draw].sum()/sizes[draw].sum()
    result={'difference':float(d.mean()),'paired_match_ci95':np.quantile(matches,[.025,.975]).tolist(),'matches':len(d),'bootstrap_repeats':repeats,'bootstrap_seed':seed}
    if groups is not None:result.update(league_cluster_ci95=np.quantile(clusters,[.025,.975]).tolist(),league_ids=len(unique))
    return result

def comparisons(pred,features,context_path):
    from aggregate_results import NEURAL,MINUTES,SEEDS
    result={'full_minus_xgboost':{},'neural_alternative_minus_full':{},'later_full_minus_comparator':{},'baseline_extension':{},'unavailable':[]}
    complete=lambda c,m,t:all((c,m,t,s) in pred for s in SEEDS)
    get=lambda c,m,t,s:pred[c,m,t,s]['probabilities']
    for t in MINUTES:
        for s in SEEDS:
            if ('historical','full',t,s) in pred and ('historical','xgboost',t,s) in pred:
                result['full_minus_xgboost'][f'{t}m_seed{s}']=paired_bootstrap(features['historical',t]['outcome'],get('historical','full',t,s),get('historical','xgboost',t,s))
    # Preserve the original single RNG stream across all 32 contrasts.
    if all(complete('historical',m,t) for t in MINUTES for m in NEURAL):
        rng=np.random.default_rng(731)
        bootstrap_order=('full','no_graph','gru',*NEURAL[3:])
        for t in MINUTES:
            y=features['historical',t]['outcome']
            for m in bootstrap_order[1:]:
                delta=np.mean([loss(y,get('historical',m,t,s),False)-loss(y,get('historical','full',t,s),False) for s in SEEDS],axis=0)
                acc=np.mean([((get('historical',m,t,s)>=.5)==y).astype(float)-((get('historical','full',t,s)>=.5)==y).astype(float) for s in SEEDS],axis=0)
                draws=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
                result['neural_alternative_minus_full'][f'{m}_minus_full_{t}m']={'log_loss_difference':float(delta.mean()),'accuracy_difference':float(acc.mean()),'paired_match_ci95':np.quantile(draws,[.025,.975]).tolist(),'bootstrap_repeats':2000,'bootstrap_seed':731,'rng_order':'minute then fixed neural order, shared stream'}
    else:result['unavailable'].append('Neural intervals require all 108 neural runs to preserve the frozen RNG order.')
    context=json.loads(context_path.read_text(encoding='utf-8'))
    rows=context['matches'];by_id={r['match_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate later group IDs')
    for t in MINUTES:
        f=features['later',t]
        if any(int(mid) not in by_id for mid in f['match_id']):raise ValueError('Missing later league metadata')
        months=np.array([by_id[int(mid)]['period'] for mid in f['match_id']])
        dates=np.array([datetime.fromtimestamp(int(ts),timezone.utc).strftime('%Y-%m') for ts in f['start_time']])
        if not np.array_equal(months,dates):raise ValueError('Later month metadata mismatch')
        groups=np.array([by_id[int(mid)]['league_id'] for mid in f['match_id']])
        for m in ('no_graph','temporal_only','gold_logistic'):
            if complete('later','full',t) and complete('later',m,t):
                y=f['outcome']
                d=np.mean([loss(y,get('later','full',t,s)) for s in SEEDS],axis=0)-np.mean([loss(y,get('later',m,t,s)) for s in SEEDS],axis=0)
                r=interval(d,groups)
                r['monthly']={month:{'n':int((months==month).sum()),'difference':float(d[months==month].mean())} for month in sorted(set(months))}
                result['later_full_minus_comparator'][f'full_minus_{m}_{t}m']=r
        for c in ('historical','later'):
            y=features[c,t]['outcome']
            for ref in ('full','no_graph'):
                for m in ('full_feature_lr','lightgbm'):
                    if complete(c,ref,t) and complete(c,m,t):
                        d=np.mean([loss(y,get(c,ref,t,s)) for s in SEEDS],axis=0)-np.mean([loss(y,get(c,m,t,s)) for s in SEEDS],axis=0)
                        result['baseline_extension'][f'{c}_{ref}_minus_{m}_{t}m']=interval(d,groups if c=='later' else None)
    result['interpretation']='Exploratory, previously inspected cohorts; percentile intervals conditional on fitted models; no multiplicity correction; no training-set uncertainty. Losses, not probabilities, averaged within match across seeds.'
    return result
