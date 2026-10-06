"""Measure neural forward-pass cost from supplied checkpoints; never train."""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

from data_protocol import read_json, write_json
from experiment import PrefixDataset
from models import DotaMultiModalPredictor, VARIANTS
from release_common import sha


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def measure(model, batch, device, warmup, repetitions):
    samples = []
    model.eval()
    with torch.inference_mode():
        for i in range(warmup + repetitions):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = 1000 * (time.perf_counter() - start)
            if i >= warmup:
                samples.append(elapsed)
    return {'median_ms': float(np.median(samples)),
            'p95_ms': float(np.percentile(samples, 95)), 'measurements_ms': samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True, help='directory containing MODEL_MINUTEm_seedSEED folders')
    p.add_argument('--features', type=Path, required=True, help='explicit complete-prefix NPZ, normally inputs/historical/test_30m.npz')
    p.add_argument('--output', type=Path, required=True, help='new directory; never overwrites a measurement')
    p.add_argument('--models', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--minute', type=int, choices=[10,20,30,40], default=30)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--devices', nargs='+', choices=['cpu','cuda'], default=['cpu'])
    p.add_argument('--batch-sizes', nargs='+', type=positive, default=[1,64])
    p.add_argument('--threads', type=positive, default=4)
    p.add_argument('--warmup', type=positive, default=10)
    p.add_argument('--repetitions', type=positive, default=50)
    args = p.parse_args()
    if args.output.exists():
        p.error('output directory already exists; choose a new directory')
    for field in ('models','devices','batch_sizes'):
        values = getattr(args,field)
        if len(values) != len(set(values)):
            p.error('duplicate entries in '+field)
    if 'cuda' in args.devices and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable; use --devices cpu')
    dataset = PrefixDataset(args.features)
    if dataset.arrays['sequence'].shape[1] != args.minute + 1:
        p.error('feature prefix length does not match --minute')
    if len(dataset) < max(args.batch_sizes):
        p.error('fewer feature rows than the requested batch size')
    checkpoints = []
    for name in args.models:
        folder = args.runs / f'{name}_{args.minute}m_seed{args.seed}'
        weight = folder/'model.pt'
        result = read_json(folder/'result.json')
        if (result['model'],result['minute'],result['seed']) != (name,args.minute,args.seed):
            p.error('checkpoint result identity mismatch for '+name)
        if not weight.is_file():
            p.error('missing checkpoint for '+name)
        checkpoints.append((name,weight))
    torch.set_num_threads(args.threads)
    protocol = {
        'purpose':'descriptive computational comparison, no predictive model selection',
        'minute':args.minute,'seed':args.seed,'neural_variants':args.models,
        'devices':args.devices,'batch_sizes':args.batch_sizes,'warmup':args.warmup,
        'repetitions':args.repetitions,'cpu_threads':torch.get_num_threads(),
        'timing_scope':'model forward only; preconstructed inputs resident on target device; excludes feature construction, transfer, disk and network',
        'output_scope':'all outputs computed by model.forward, including auxiliary head; not end-to-end service latency',
        'batch_selection':'first B rows in supplied NPZ, no resampling',
        'feature_sha256':sha(args.features),'feature_rows':len(dataset),
        'cuda_synchronization':'before start and after forward on each call',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                   'torch':torch.__version__,'cuda':torch.version.cuda,
                   'cudnn':torch.backends.cudnn.version(),'os':platform.platform(),
                   'cpu':platform.processor(),'device':torch.cuda.get_device_name(0) if 'cuda' in args.devices else 'CPU only',
                   'interop_threads':torch.get_num_interop_threads(),
                   'cudnn_benchmark':torch.backends.cudnn.benchmark,
                   'cudnn_deterministic':torch.backends.cudnn.deterministic,
                   'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32},
        'source_sha256':{name:sha(Path(__file__).parent/name) for name in ('benchmark_complexity.py','models.py','experiment.py')},
        'limitations':'one ordered pass on one machine; no hardware exclusivity or latency confidence interval; p95 is an empirical call-time percentile; FLOPs and training cost not measured',
    }
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',protocol)
    rows=[]
    for name,weight in checkpoints:
        saved=torch.load(weight,map_location='cpu',weights_only=True)
        model=DotaMultiModalPredictor(**saved['model_config'])
        model.load_state_dict(saved['state_dict'])
        components={}
        for key,param in model.named_parameters():
            if param.requires_grad:
                branch=key.split('.')[0]
                components[branch]=components.get(branch,0)+param.numel()
        row={'model':name,'parameters':sum(components.values()),'parameter_components':components,
             'checkpoint_sha256':sha(weight),'timings':[]}
        for device_name in args.devices:
            device=torch.device(device_name);model=model.to(device)
            for size in args.batch_sizes:
                batch={key:torch.from_numpy(value[:size]).to(device) for key,value in dataset.arrays.items()}
                batch['lengths']=torch.full((size,),args.minute+1,dtype=torch.long,device=device)
                row['timings'].append({'device':device_name,'batch_size':size,
                                      **measure(model,batch,device,args.warmup,args.repetitions)})
        rows.append(row)
        # Interrupted output is visibly incomplete and rejected by the renderer.
        write_json(args.output/'measurements.json',{'status':'running','protocol':protocol,'neural':rows})
        print(name,row['parameters'],flush=True)
    write_json(args.output/'measurements.json',{'status':'complete','protocol':protocol,'neural':rows})
    print('Complete:',args.output/'measurements.json')


if __name__=='__main__':main()
