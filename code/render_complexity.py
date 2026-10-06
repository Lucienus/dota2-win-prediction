"""Regenerate timing tables from recorded measurements without loading models."""
import argparse
import json
import math
import hashlib
from pathlib import Path
import numpy as np

NAMES={'full':'Full model','gru':'GRU','no_graph':'Node MLP','no_role':'No resource weighting',
       'complete_graph':'Complete graph','no_draft_branch':'No draft branch','mean_draft':'Mean draft',
       'single_task':'Single task','temporal_only':'Temporal only'}


def validate(data):
    if data.get('status','complete')!='complete':
        raise ValueError('Incomplete measurement run')
    protocol=data['protocol'];rows=data['neural']
    if [r['model'] for r in rows]!=protocol['neural_variants']:
        raise ValueError('Model order or model coverage mismatch')
    devices=protocol.get('devices',['cpu','cuda'])
    expected={(d,b) for d in devices for b in protocol['batch_sizes']}
    if len(expected)!=len(devices)*len(protocol['batch_sizes']):
        raise ValueError('Duplicate conditions in protocol')
    if len({r['model'] for r in rows})!=len(rows):
        raise ValueError('Duplicate model')
    for row in rows:
        if row['model'] not in NAMES:raise ValueError('Unknown model')
        if sum(row['parameter_components'].values())!=row['parameters']:
            raise ValueError('Parameter component sum mismatch')
        actual=[(t['device'],t['batch_size']) for t in row['timings']]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:
            raise ValueError('Timing condition coverage mismatch')
        for timing in row['timings']:
            samples=timing['measurements_ms']
            if len(samples)!=protocol['repetitions'] or not samples:
                raise ValueError('Wrong timed sample count')
            if not all(math.isfinite(x) and x>0 for x in samples):
                raise ValueError('Nonfinite/nonpositive timing')
            for key,value in [('median_ms',float(np.median(samples))),('p95_ms',float(np.percentile(samples,95)))]:
                if not math.isclose(value,timing[key],rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Summary does not match raw samples: '+key)
    return [(d,b) for d in devices for b in protocol['batch_sizes']]


def tables(data):
    conditions=validate(data);p=data['protocol']
    columns=[f'{d.upper()}, batch {b}' for d,b in conditions]
    md=['| Configuration | Parameters | '+' | '.join(columns)+' |',
        '|---|---:|'+'---:|'*len(columns)]
    tex=[r'\begin{table*}[t]',r'\caption{Median forward-pass time in milliseconds per batch. Minute '+str(p['minute'])+
         ', seed '+str(p['seed'])+', '+str(p['cpu_threads'])+' CPU threads; '+str(p['warmup'])+' warm-up and '+str(p['repetitions'])+
         r' timed calls per condition. Inputs are resident on the target device; construction, transfers and I/O are excluded. Hardware and raw timings are recorded in the accompanying measurement JSON.}',
         r'\label{tab:latency_remeasurement}\centering\small',r'\begin{tabular}{@{}lr'+'r'*len(columns)+r'@{}}\toprule',
         'Configuration & Parameters & '+' & '.join(columns)+r'\\\midrule']
    for row in data['neural']:
        lookup={(t['device'],t['batch_size']):t['median_ms'] for t in row['timings']}
        values=[f'{lookup[c]:.3f}' for c in conditions]
        md.append('| '+NAMES[row['model']]+f" | {row['parameters']:,} | "+' | '.join(values)+' |')
        tex.append(NAMES[row['model']]+' & '+str(row['parameters'])+' & '+' & '.join(values)+r' \\')
    tex.append(r'\bottomrule\end{tabular}\end{table*}')
    md+=['','Conditions and limitations:',json.dumps(p,ensure_ascii=False,indent=2),
         '', 'p95 is a percentile of timed calls, not a confidence interval. New hardware timings do not replace frozen manuscript results automatically.']
    return '\n'.join(tex)+'\n','\n'.join(md)+'\n'


def paper_reference(path,data):
    validate(data)
    reference=Path(__file__).resolve().parents[1]/'benchmark_reference'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=hashlib.sha256((reference/'measurements.json').read_bytes()).hexdigest():
        raise ValueError('--paper-reference requires the bundled frozen measurement file; use the default renderer for new measurements')
    text=(reference/'paper_timing.template.tex').read_text('utf-8')
    for row in data['neural']:
        for timing in row['timings']:
            token='@@'+row['model']+'_'+timing['device']+'_'+str(timing['batch_size'])+'@@'
            text=text.replace(token,f"{timing['median_ms']:.3f}")
    if '@@' in text:raise ValueError('Unresolved table template field')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='new output directory')
    p.add_argument('--paper-reference',action='store_true',help='exact frozen manuscript table and accompanying paragraph, using bundled reference measurements')
    args=p.parse_args();data=json.loads(args.input.read_text('utf-8'))
    tex,md=tables(data)
    if args.paper_reference:tex=paper_reference(args.input,data)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'complexity_timing.tex').write_text(tex,encoding='utf-8')
    (args.output/'complexity_report.md').write_text(md,encoding='utf-8')
    (args.output/'source.json').write_text(json.dumps({'measurement_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
      'mode':'frozen_paper_reference' if args.paper_reference else 'measurement_table'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
