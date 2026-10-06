"""Paper-shaped tabular fragments; missing seed cells are never silently averaged."""
import numpy as np
from evaluation import probability_metrics

LABELS=dict(full='Full model',gold_logistic='Gold/XP LR',random_forest='Random forest',xgboost='XGBoost',gru='GRU',no_graph='Node MLP',no_role='No resource weighting',complete_graph='Complete graph',no_draft_branch='No draft branch',mean_draft='Mean draft',single_task='Single task',temporal_only='Temporal only',full_feature_lr='Full-feature LR',lightgbm='LightGBM')

def render_tables(out,summary,features,params,pred,contrasts,inputs):
    from aggregate_results import NEURAL,MINUTES,SEEDS,METRICS
    cells={(r['cohort'],r['model'],r['minute']):r for r in summary};inventory={}
    def table(name,headers,rows):
        text='\\begin{tabular}{@{}'+'l'*len(headers)+'@{}}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
        text+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
        (out/(name+'.tex')).write_text(text,encoding='utf-8')
        (out/(name+'.md')).write_text('| '+' | '.join(headers)+' |\n|'+'---|'*len(headers)+'\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n',encoding='utf-8')
        inventory[name]={'rows':len(rows),'contains_missing':any('--'==str(v) for r in rows for v in r)}
    def cell(m,t,c='historical'):return cells.get((c,m,t))
    def value(m,t,k='accuracy',c='historical',sd=False):
        r=cell(m,t,c)
        if not r or not r['complete'] or r[k+'_mean'] is None:return '--'
        if sd:return f"${100*r[k+'_mean']:.2f} \\pm {100*r[k+'_sd']:.2f}$"
        return f"{r[k+'_mean']:.4f}"
    n=lambda c,t:f"{len(features[c,t]['match_id']):,}"
    base=('full','gold_logistic','random_forest','xgboost')
    headers=['Model',*[f'{t} min' for t in MINUTES]]
    table('accuracy',headers,[[LABELS[m]]+[value(m,t,sd=True) for t in MINUTES] for m in base]+[['Test matches']+[n('historical',t) for t in MINUTES]])
    table('probability',['Minute','Model','Bal. acc.','F1','AUC','Log loss','Brier','ECE'],[[t,LABELS[m]]+[value(m,t,k) for k in METRICS[1:]] for t in MINUTES for m in base])
    table('ablation_accuracy',['Configuration','Parameters',*headers[1:]],[[LABELS[m],f'{params[m]:,}' if m in params else '--']+[value(m,t,sd=True) for t in MINUTES] for m in NEURAL])
    table('neural_complexity_ll',['Configuration','Parameters',*headers[1:]],[[LABELS[m],f'{params[m]:,}' if m in params else '--']+[value(m,t,'log_loss') for t in MINUTES] for m in NEURAL])
    table('ablation_probability',['Configuration',*[f'{k} {t}' for t in MINUTES for k in ('LL','Brier')]],[[LABELS[m]]+[value(m,t,k) for t in MINUTES for k in ('log_loss','brier')] for m in NEURAL])
    table('external_results',['Minute','Model','n','Accuracy (\\%)','Log loss','Brier','AUC','ECE'],[[t,LABELS[m],n('later',t),value(m,t,c='later',sd=True)]+[value(m,t,k,'later') for k in ('log_loss','brier','roc_auc','ece_10_equal_width_bins')] for t in MINUTES for m in ('full','no_graph','temporal_only','gold_logistic')])
    table('baseline_extension',['Cohort','Minute','Matches','Full','Node MLP','XGBoost','Full-feature LR','LightGBM'],[[c.title(),t,n(c,t)]+[value(m,t,'log_loss',c) for m in ('full','no_graph','xgboost','full_feature_lr','lightgbm')] for c in ('historical','later') for t in MINUTES])
    rows=[]
    for t in MINUTES:
        for s in SEEDS:
            r=contrasts['full_minus_xgboost'].get(f'{t}m_seed{s}')
            if r:
                a,b=r['accuracy'],r['brier']
                rows.append([t,s,f"${100*a['difference_a_minus_b']:+.2f}$",f"$[{100*a['ci95'][0]:+.2f}, {100*a['ci95'][1]:+.2f}]$",f"${b['difference_a_minus_b']:+.4f}$",f"$[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]$"])
            else:rows.append([t,s,*(['--']*4)])
    table('paired',['Minute','Seed','Acc. diff.','95\\% interval','Brier diff.','95\\% interval'],rows)
    rows=[]
    for t in MINUTES:
        f=features['historical',t]
        for label,mask in [('Patch 59',f['patch']==59),('$|G_t|\\leq1{,}000$',np.abs(f['gold_difference'])<=.1)]:
            row=[label,t,f'{mask.sum():,}']
            for m in base:
                r=cell(m,t)
                row.append(f"{100*np.mean([probability_metrics(f['outcome'][mask],pred['historical',m,t,s]['probabilities'][mask])['accuracy'] for s in SEEDS]):.2f}" if r and r['complete'] and mask.any() else '--')
            rows.append(row)
    table('subsets',['Subset','Minute','Matches','Full','Gold/XP LR','RF','XGBoost'],rows)
    rows=[]
    for t in MINUTES:
        counts=[]
        for split in ('train','validation','test'):
            with np.load(inputs/'historical'/f'{split}_{t}m.npz',allow_pickle=False) as z:counts.append(f"{len(z['match_id']):,}")
        rows.append([t,*counts])
    table('cohorts',['Minute','Training','Validation','Test'],rows)
    for name,block in [('neural_paired',contrasts['neural_alternative_minus_full']),('later_paired',contrasts['later_full_minus_comparator']),('baseline_paired',contrasts['baseline_extension'])]:
        rows=[]
        for key,r in block.items():
            key=key.replace('_',r'\_');d=r.get('difference',r.get('log_loss_difference'))
            ci=lambda vals:'['+', '.join(f'{x:+.6f}' for x in vals)+']'
            rows.append([key,f'{d:+.6f}',ci(r['paired_match_ci95']),ci(r['league_cluster_ci95']) if 'league_cluster_ci95' in r else '--'])
        table(name,['Comparison','Log loss difference','Match 95\\% interval','League 95\\% interval'],rows)
    table('later_monthly',['Comparison','Month','Matches','Difference'],[[key.replace('_',r'\_'),month,r['n'],f"{r['difference']:+.6f}"] for key,v in contrasts['later_full_minus_comparator'].items() for month,r in v['monthly'].items()])
    macros=[]
    for t,word in zip(MINUTES,('Ten','Twenty','Thirty','Forty')):
        r=cell('full',t);v=f"{100*r['accuracy_mean']:.2f}" if r and r['complete'] else '--'
        macros.append(f'\\newcommand{{\\Acc{word}}}{{{v}}}')
    r=cell('xgboost',30);v=f"{100*r['accuracy_mean']:.2f}" if r and r['complete'] else '--'
    macros.append(f'\\newcommand{{\\XgbThirty}}{{{v}}}')
    (out/'result_macros.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
    inventory['excluded_static_evidence']=['competition_scope','external_cohorts','sensitivity','timing macros','architecture/feature-description tables']
    return inventory
