"""Verify a downloaded release against its file checksums."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'checksums.json').read_text('utf-8'))
for name,expected in manifest.items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe manifest path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    if h.hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
print('Verified',len(manifest),'release files')
