"""Rebuild corrected features directly from raw JSON, without legacy caches."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from data_protocol import ROOT, load_manifest, write_json, fingerprint
from feature_extractor import fit_purchase_prior, feature_contract
from v4_objectives import TowerCorrectedExtractor
from release_common import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-cache',type=Path)
    args=p.parse_args()
    manifest=load_manifest(args.manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    contract={'manifest_fingerprint':manifest['fingerprint'],'feature_contract':feature_contract(),
              'source_hashes':{x:sha(ROOT/x) for x in ('feature_extractor.py','v4_objectives.py','prepare_raw.py')},
              'aux_horizon':5,'minutes':[10,20,30,40],'debug_subset':bool(manifest.get('debug_subset'))}
    def read(row):
        path=(args.raw_dir/row['file']).resolve()
        if not path.is_relative_to(args.raw_dir.resolve()):raise ValueError('Unsafe raw path')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Changed raw match: '+row['file'])
        return json.loads(raw)
    # Verify all train bytes before fitting the train-only purchase prior.
    for row in manifest['splits']['train']:read(row)
    prior=fit_purchase_prior(manifest['splits']['train'],args.raw_dir,manifest['fingerprint'])
    write_json(args.output/'purchase_prior.json',prior)
    comparisons={};counts={}
    if args.compare_cache:
        expected=json.loads((args.compare_cache/'purchase_prior.json').read_text('utf-8'))
        if prior!=expected:raise ValueError('Purchase prior differs from frozen prior')
    for split,rows in manifest['splits'].items():
        groups={m:[] for m in (10,20,30,40)}
        for i,row in enumerate(rows):
            match=read(row)
            for minute in groups:
                if row['duration']>60*minute and row['max_prefix_minute']>=minute:
                    sample=TowerCorrectedExtractor(match,minute,prior,5).extract()
                    sample['patch']=np.int64(row['patch'] if isinstance(row.get('patch'),int) else -1)
                    sample['start_time']=np.int64(row['start_time'])
                    groups[minute].append(sample)
            if (i+1)%2000==0:print(split,i+1,len(rows),flush=True)
        for minute,samples in groups.items():
            arrays={k:np.stack([s[k] for s in samples]) for k in samples[0]}
            if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite features')
            name=f'{split}_{minute}m.npz'
            np.savez_compressed(args.output/name,**arrays)
            counts[name]=len(samples)
            if args.compare_cache:
                with np.load(args.compare_cache/name,allow_pickle=False) as old:
                    check={k:bool(k in old and arrays[k].dtype==old[k].dtype and np.array_equal(arrays[k],old[k])) for k in arrays}
                    if set(old.files)!=set(arrays) or not all(check.values()):raise ValueError(f'Frozen array mismatch: {name}: {check}')
                    comparisons[name]=check
        print('Finished',split,flush=True)
    files={f.name:sha(f) for f in args.output.iterdir() if f.suffix in ('.npz','.json')}
    write_json(args.output/'metadata.json',{'cache_id':fingerprint(contract),'contract':contract,'counts':counts,'files':files,
                                        'frozen_array_comparisons':comparisons,'identity':'new direct-extraction release build'})
    print('COMPLETE: corrected raw-to-feature rebuild',counts,flush=True)

if __name__=='__main__':main()
