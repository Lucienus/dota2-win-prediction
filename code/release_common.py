"""Portable input verification. Original cache identity is preserved, never forged."""
import hashlib
from pathlib import Path
from data_protocol import read_json

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_cache(folder):
    folder=Path(folder)
    meta=read_json(folder/'metadata.json')
    for name,expected in meta['files'].items():
        target=(folder/name).resolve()
        if not target.is_relative_to(folder.resolve()):raise ValueError('Unsafe cache path')
        if sha(target)!=expected:raise ValueError('Input hash mismatch: '+name)
    return meta
