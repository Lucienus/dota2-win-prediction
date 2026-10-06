"""Validate new fit outputs and generate separate, prediction-derived paper tables."""
import argparse
import csv
import json
from itertools import product
from pathlib import Path
import numpy as np
from evaluation import probability_metrics
from release_common import sha, load_cache
from later_scope import PAPER_MODELS, PAPER_MINUTES, PAPER_SEEDS

ROOT = Path(__file__).resolve().parents[1]
NEURAL = ('full','gru','no_graph','no_role','complete_graph','no_draft_branch','mean_draft','single_task','temporal_only')
MODELS = NEURAL + ('gold_logistic','random_forest','xgboost','full_feature_lr','lightgbm')
METRICS = ('accuracy','balanced_accuracy','f1','roc_auc','log_loss','brier','ece_10_equal_width_bins')
MINUTES, SEEDS = PAPER_MINUTES, PAPER_SEEDS

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def expected_grid():
    return set(('historical',m,t,s) for m,t,s in product(MODELS,MINUTES,SEEDS)) | set(('later',m,t,s) for m,t,s in product(PAPER_MODELS,MINUTES,SEEDS))

def validate_predictions(path, ids, labels):
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('match_id','labels','probabilities')}
    if any(v.ndim!=1 for v in a.values()):raise ValueError('Prediction vectors must be one-dimensional: '+str(path))
    if len(np.unique(a['match_id']))!=len(a['match_id']):raise ValueError('Duplicate match IDs: '+str(path))
    if not np.array_equal(a['match_id'],ids) or not np.array_equal(a['labels'],labels):
        raise ValueError('Prediction IDs/order/labels differ from frozen cohort: '+str(path))
    if not np.isin(a['labels'],[0,1]).all():raise ValueError('Non-binary labels')
    metrics=probability_metrics(a['labels'],a['probabilities'])
    return a,metrics

def check_metrics(actual,saved):
    for k in ('n',*METRICS):
        if k not in saved:raise ValueError('Missing saved metric: '+k)
        a,b=actual[k],saved[k]
        if (a is None)!=(b is None) or (a is not None and not np.isclose(a,b,atol=1e-12,rtol=0)):
            raise ValueError('Saved metric differs from predictions: '+k)

def summarize(records):
    summary=[]
    for cohort,model,minute in sorted({(r['cohort'],r['model'],r['minute']) for r in records}):
        group=sorted([r for r in records if (r['cohort'],r['model'],r['minute'])==(cohort,model,minute)],key=lambda r:r['seed'])
        seeds=[r['seed'] for r in group]
        row={'cohort':cohort,'model':model,'minute':minute,'n':group[0]['n'],'seeds':seeds,'seed_count':len(seeds),'complete':seeds==list(SEEDS)}
        for metric in METRICS:
            values=[r[metric] for r in group]
            defined=all(v is not None for v in values)
            row[metric+'_mean']=float(np.mean(values)) if defined else None
            row[metric+'_sd']=float(np.std(values,ddof=1)) if defined and len(values)>1 else None
        summary.append(row)
    return summary

def collect(runs, later_runs, inputs):
    """Read only. Never load pickled checkpoints or choose runs using test scores."""
    runs,inputs=Path(runs).resolve(),Path(inputs).resolve()
    if not runs.is_dir():raise ValueError('Training directory does not exist')
    meta=load_cache(inputs/'historical')
    later_meta=read(inputs/'later/metadata.json')
    for name,digest in later_meta['feature_files'].items():
        if sha(inputs/'later'/Path(name).name)!=digest:raise ValueError('Later feature hash mismatch')
    features={};feature_hashes={}
    for cohort in ('historical','later'):
        for minute in MINUTES:
            p=inputs/cohort/(f'test_{minute}m.npz' if cohort=='historical' else f'{minute}m.npz')
            with np.load(p,allow_pickle=False) as z:
                features[cohort,minute]={k:z[k].copy() for k in ('match_id','outcome','patch','start_time')}
                features[cohort,minute]['gold_difference']=z['sequence'][:,-1,0].copy()
            feature_hashes[p.relative_to(inputs).as_posix()]=sha(p)
    records=[];predictions={};manifest=[];seen=set();weights={};configs={};parameters={};pending=[]
    def add(path,cohort,result):
        key=(cohort,result['model'],result['minute'],result['seed'])
        if key not in expected_grid():raise ValueError('Unexpected paper configuration: '+str(key))
        if key in seen:raise ValueError('Duplicate configuration: '+str(key))
        seen.add(key)
        context=features[cohort,key[2]]
        pp=path.parent/'predictions.npz'
        pred,metrics=validate_predictions(pp,context['match_id'],context['outcome'])
        check_metrics(metrics,result['test'] if cohort=='historical' else result['metrics'])
        records.append(dict(zip(('cohort','model','minute','seed'),key))|{k:metrics[k] for k in ('n',*METRICS)})
        predictions[key]=pred
        manifest.append({'cohort':cohort,'model':key[1],'minute':key[2],'seed':key[3],
                         'result_path':str(path),'result_sha256':sha(path),'predictions_path':str(pp),'predictions_sha256':sha(pp)})
        return key
    for folder in sorted(runs.iterdir()):
        if not folder.is_dir():continue
        p=folder/'result.json'
        if not p.exists():
            if (folder/'run_config.json').exists():pending.append(folder.name)
            continue
        r=read(p);spec=read(folder/'run_config.json')
        if spec!=r.get('specification'):raise ValueError('Saved run/result specification mismatch: '+str(folder))
        if any(spec.get(k)!=r[k] for k in ('model','minute','seed')):raise ValueError('Run identity mismatch')
        if spec.get('cache_files')!=meta['files']:raise ValueError('Training feature contract differs')
        key=add(p,'historical',r)
        candidates=[folder/n for n in ('model.pt','model.joblib') if (folder/n).is_file()]
        if len(candidates)!=1:raise ValueError('Expected one selected checkpoint: '+str(folder))
        weights[key[1:]]=sha(candidates[0])
        manifest[-1]['checkpoint_sha256']=weights[key[1:]]
        manifest[-1]['specification']=spec
        # A three-seed cell must not combine different code, data, budgets or runtimes.
        comparable={k:v for k,v in spec.items() if k!='seed'}
        cell=key[1:3]
        if cell in configs and configs[cell]!=comparable:raise ValueError('Mixed specifications within seed cell: '+str(cell))
        configs[cell]=comparable
        if r['model'] in NEURAL:
            count=r.get('parameters')
            if type(count)!=int or count<=0:raise ValueError('Missing neural parameter count')
            if r['model'] in parameters and parameters[r['model']]!=count:raise ValueError('Inconsistent parameter counts')
            parameters[r['model']]=count
    if not records:raise ValueError('No completed training runs')
    if later_runs is not None:
        later_runs=Path(later_runs).resolve()
        if not later_runs.is_dir():raise ValueError('Later results directory does not exist')
        audit=read(later_runs/'evaluation_scope.json')
        declared={(r['model'],r['minute'],r['seed']) for r in audit['selected']}
        if len(declared)!=audit['selected_count'] or len(audit['selected'])!=len(declared):raise ValueError('Invalid later selection audit')
        found=set()
        for p in sorted(later_runs.glob('*/result.json')):
            r=read(p);identity=(r['model'],r['minute'],r['seed'])
            if r.get('evaluation_scope')!=audit['scope']:raise ValueError('Later evaluation scope mismatch')
            if identity not in weights or r.get('checkpoint_sha256')!=weights[identity]:raise ValueError('Later result uses a different/missing training checkpoint')
            add(p,'later',r);found.add(identity)
        if found!=declared:raise ValueError('Later results incomplete or different from selection audit')
    return records,predictions,features,parameters,manifest,pending,feature_hashes

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--later-results',type=Path)
    p.add_argument('--inputs',type=Path,default=ROOT/'inputs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Diagnostic summaries only; incomplete cells never become paper means')
    p.add_argument('--dry-run',action='store_true',help='Validate predictions and print completeness without writing tables')
    args=p.parse_args(argv)
    if args.output.exists():raise ValueError('Output must be a new directory')
    protected=[ROOT,args.runs.resolve(),args.inputs.resolve()]
    if args.later_results:protected.append(args.later_results.resolve())
    if any(args.output.resolve().is_relative_to(r) for r in protected):raise ValueError('Output must be outside release/input/run directories')
    records,pred,features,params,manifest,pending,hashes=collect(args.runs,args.later_results,args.inputs)
    actual={(r['cohort'],r['model'],r['minute'],r['seed']) for r in records}
    missing=sorted(expected_grid()-actual)
    coverage={'complete':not missing,'expected_runs':252,'completed_runs':len(records),'missing':[dict(zip(('cohort','model','minute','seed'),k)) for k in missing],'unfinished_directories':pending}
    print(json.dumps(coverage if args.dry_run else {k:v for k,v in coverage.items() if k!='missing'}|{'missing_count':len(missing)},indent=2))
    if missing and not args.allow_partial:raise ValueError('Incomplete paper grid; no outputs written. Use --allow-partial only for a separate diagnostic report.')
    if args.dry_run:return
    from aggregate_statistics import comparisons
    from aggregate_tables import render_tables
    summary=summarize(records)
    contrasts=comparisons(pred,features,ROOT/'analysis_context/later_groups.json')
    # All validation/statistical computation succeeds before creating output.
    args.output.mkdir(parents=True,exist_ok=False)
    tables=args.output/'tables';tables.mkdir()
    write(args.output/'coverage.json',coverage)
    write(args.output/'summary.json',summary)
    csv_write(args.output/'metrics.csv',records);csv_write(args.output/'summary.csv',summary)
    write(args.output/'comparisons.json',contrasts)
    write(args.output/'runs_manifest.json',{'schema':1,'source_kind':'supplied_training_outputs','records':manifest,'input_sha256':hashes,'analysis_code_sha256':{n:sha(Path(__file__).parent/n) for n in ('aggregate_results.py','aggregate_statistics.py','aggregate_tables.py','evaluation.py')},'analysis_context_sha256':sha(ROOT/'analysis_context/later_groups.json'),'numpy':np.__version__})
    generated=render_tables(tables,summary,features,params,pred,contrasts,args.inputs)
    write(args.output/'table_inventory.json',generated)
    lines=['# 新运行结果汇总','',f"已核验 {len(records)}/252 个运行。完整论文网格：{not missing}。",'',
           '逐场预测重新计算指标；n为比赛数。三个种子的指标分别计算后取均值，SD为样本标准差(ddof=1)，不把种子当成新增比赛。',
           '单种子SD留空；不完整单元在论文表格中显示--，已观察均值仅保留于诊断summary文件。',
           '区间以比赛配对，损失先在每场内跨种子平均；联赛区间是依赖敏感性分析。全部比较为已查看队列上的探索性分析，未作多重比较校正，不自动生成显著性或优越性结论。',
           'tables目录为新生成结果；原论文、冻结结果及旧权重不作修改。静态数据审计、架构说明、单场敏感性及原硬件计时不由预测汇总推算。计时请使用benchmark_complexity.py与render_complexity.py。',
           'runs_manifest.json保留来源绝对路径用于本地追溯，公开此报告前须检查这些路径。', '', '## 当前结果（只有三个种子齐全才显示论文均值）','',
           '| 队列 | 模型 | 分钟 | 比赛 | 种子数 | 准确率 | Log loss |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        acc=f"{100*r['accuracy_mean']:.2f}%" if r['complete'] else '待齐全'
        ll=f"{r['log_loss_mean']:.4f}" if r['complete'] else '待齐全'
        lines.append(f"| {r['cohort']} | {r['model']} | {r['minute']} | {r['n']} | {r['seed_count']}/3 | {acc} | {ll} |")
    (args.output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(args.output/'checksums.json',{f.relative_to(args.output).as_posix():sha(f) for f in sorted(args.output.rglob('*')) if f.is_file()})
    print('Generated separate result tables:',args.output)

if __name__=='__main__':main()
