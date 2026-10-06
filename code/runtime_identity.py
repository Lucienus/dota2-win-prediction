"""Stable, actual imported dependency identity and readable resume differences."""
import importlib
import json
import platform
from pathlib import Path

DEPENDENCIES = ('numpy','scipy','sklearn','torch','xgboost','lightgbm','joblib','threadpoolctl')


def dependency_versions():
    versions={}
    for name in DEPENDENCIES:
        try:
            module=importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name!=name:
                raise RuntimeError('Broken dependency import: '+name) from exc
            versions[name]=None
        else:
            version=getattr(module,'__version__',None)
            if version is None:
                raise RuntimeError('Cannot identify imported dependency version: '+name)
            versions[name]=str(version)
    return versions


def collect_environment():
    # Import all tracked packages before enumerating loaded BLAS/OpenMP runtimes,
    # so inventory does not depend on which model family imported them first.
    versions=dependency_versions()
    import torch
    from threadpoolctl import threadpool_info
    pools=[]
    for pool in threadpool_info():
        pools.append({k:pool.get(k) for k in ('user_api','internal_api','prefix','version',
                                             'num_threads','threading_layer','architecture')})
    pools.sort(key=lambda x:json.dumps(x,sort_keys=True))
    cuda=torch.cuda.is_available()
    return {'schema_version':2,'python':platform.python_version(),
            'python_implementation':platform.python_implementation(),
            'platform':platform.platform(),'machine':platform.machine(),
            'processor':platform.processor(),'dependencies':versions,
            'cuda_available':cuda,'cuda_runtime':torch.version.cuda,
            'cudnn_version':torch.backends.cudnn.version(),
            'cuda_devices':[{'name':torch.cuda.get_device_name(i),
                             'capability':list(torch.cuda.get_device_capability(i))}
                            for i in range(torch.cuda.device_count())] if cuda else [],
            'torch_num_threads':torch.get_num_threads(),
            'torch_interop_threads':torch.get_num_interop_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'threadpools':pools}


def differences(saved,current,prefix=''):
    """Return changed field names/values without timestamps or absolute paths."""
    if isinstance(saved,dict) and isinstance(current,dict):
        out=[]
        for key in sorted(saved.keys()|current.keys()):
            path=prefix+'.'+key if prefix else key
            if key not in saved:out.append(path+': missing in saved configuration')
            elif key not in current:out.append(path+': absent from current configuration')
            else:out.extend(differences(saved[key],current[key],path))
        return out
    return [] if saved==current else [prefix+': '+repr(saved)+' -> '+repr(current)]


def assert_same(saved,current,context):
    changes=differences(saved,current)
    if changes:
        detail='\n'.join(changes[:12])
        raise ValueError(f'Cannot resume {context}: configuration/runtime changed.\n{detail}\n'
                         'Use a new output directory. Do not edit the old identity to bypass this check.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(collect_environment(),stream,indent=2)
