"""Rebuild frozen cohorts; default rejects changed raw bytes, never resamples."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from data_protocol import read_json,write_json,fingerprint
from feature_extractor import fit_purchase_prior
from v4_objectives import TowerCorrectedExtractor
from release_common import sha
from fetch_frozen_matches import rows_from,validate
ROOT=Path(__file__).resolve().parents[1]
def fields(arrays):
    return {k:{'dtype':v.dtype.str,'shape':list(v.shape),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in arrays.items()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=['historical','later'],required=True);p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior',type=Path);p.add_argument('--allow-changed-raw',action='store_true');a=p.parse_args()
    m=read_json(ROOT/f'manifests/{a.cohort}.json');rows=rows_from(m);lookup={r['match_id']:r for r in rows}
    if a.cohort=='later' and not a.prior:p.error('Later extraction requires --prior from rebuilt historical training data')
    if a.output.exists():p.error('Output must be a new directory')
    drift=[]
    for r in rows:
        path=a.raw_dir/r['file'];raw=path.read_bytes();validate(raw,r['match_id'])
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:drift.append(r['match_id'])
    if drift and not a.allow_changed_raw:raise ValueError(f'{len(drift)} raw hashes differ; original files or explicit --allow-changed-raw required')
    a.output.mkdir(parents=True)
    write_json(a.output/'rebuild_status.json',{'complete':False,'raw_hash_mismatches':drift,'changed_raw_allowed':a.allow_changed_raw})
    if a.cohort=='historical':
        prior=fit_purchase_prior(m['splits']['train'],a.raw_dir,m['original_split_fingerprint']);write_json(a.output/'purchase_prior.json',prior)
        jobs=[(f'{split}_{minute}m.npz',[r['match_id'] for r in group if minute in r['eligible_minutes']],minute) for split,group in m['splits'].items() for minute in [10,20,30,40]]
    else:
        prior=read_json(a.prior);jobs=[(f'{minute}m.npz',m['minute_ids'][str(minute)],minute) for minute in [10,20,30,40]]
    reference=read_json(ROOT/'manifests/feature_field_hashes.json')['arrays'];comparisons={};counts={}
    for name,ids,minute in jobs:
        samples=[]
        for mid in ids:
            raw=(a.raw_dir/lookup[mid]['file']).read_bytes();obj=validate(raw,mid)
            if hashlib.sha256(raw).hexdigest()!=lookup[mid]['sha256'] and not a.allow_changed_raw:raise ValueError('Raw changed during build')
            sample=TowerCorrectedExtractor(obj,minute,prior,5).extract()
            if a.cohort=='later':
                # The frozen evaluation-only cohort has no auxiliary training targets.
                sample['aux']=np.zeros(2,dtype=np.float32);sample['aux_mask']=np.float32(0)
            sample['patch']=np.int64(obj['patch'] if isinstance(obj.get('patch'),int) else -1);sample['start_time']=np.int64(obj['start_time']);samples.append(sample)
        arrays={k:np.stack([s[k] for s in samples]) for k in samples[0]}
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite feature')
        actual=fields(arrays);expected=reference[f'{a.cohort}/{name}'];comparisons[name]={'equal':actual==expected,'observed_fields':actual}
        np.savez_compressed(a.output/name,**arrays);counts[name]=len(ids);print(name,len(ids),comparisons[name]['equal'],flush=True)
    exact=all(r['equal'] for r in comparisons.values())
    status={'complete':True,'cohort':a.cohort,'raw_hash_mismatches':drift,'features_equal_frozen':exact,'field_comparisons':comparisons,'counts':counts,'manifest_sha256':sha(ROOT/f'manifests/{a.cohort}.json'),'source_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},'identity':'new rebuild; original run identity not reused'}
    write_json(a.output/'rebuild_status.json',status)
    if not exact and not a.allow_changed_raw:raise ValueError('Feature hashes differ; no training metadata written. See rebuild_status.json')
    if a.cohort=='historical':write_json(a.output/'metadata.json',{'cache_id':fingerprint(status),'files':{f.name:sha(f) for f in a.output.iterdir() if f.suffix in ['.npz','.json']},'counts':counts,'features_equal_frozen':exact})
    else:write_json(a.output/'metadata.json',{'feature_files':{name:sha(a.output/name) for name in counts},'counts':counts,'features_equal_frozen':exact,'rebuild_identity':fingerprint(status)})
if __name__=='__main__':main()
