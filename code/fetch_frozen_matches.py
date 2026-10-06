"""Download frozen IDs only; record errors and drift without replacing matches."""
import argparse,hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

def sha(raw): return hashlib.sha256(raw).hexdigest()
def rows_from(manifest):
    rows=[r for part in manifest['splits'].values() for r in part] if 'splits' in manifest else manifest['included']
    ids=[r['match_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate frozen match ID')
    for r in rows:
        if not isinstance(r['match_id'],int) or r['match_id']<=0 or r['file']!=str(r['match_id'])+'.json':raise ValueError('Invalid manifest identity/path')
    return rows
def validate(raw,mid):
    obj=json.loads(raw)
    if not isinstance(obj,dict) or obj.get('match_id')!=mid:raise ValueError('Response match identity differs')
    if len(obj.get('players',[]))!=10:raise ValueError('Response lacks ten players')
    return obj
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--interval',type=float,default=1.2);a=p.parse_args()
    if a.interval<1 or (a.limit is not None and a.limit<1):p.error('interval must be >=1 second and limit positive')
    rows=rows_from(json.loads(a.manifest.read_text('utf-8')))
    if a.limit:rows=rows[:a.limit]
    a.output.mkdir(parents=True,exist_ok=True);key=os.environ.get('OPENDOTA_API_KEY','');counts={}
    with (a.output/'fetch_log.jsonl').open('a',encoding='utf-8') as log:
        for r in rows:
            mid=r['match_id'];dest=a.output/r['file'];record={'match_id':mid,'expected_sha256':r['sha256'],'checked_utc':datetime.now(timezone.utc).isoformat()}
            try:
                if dest.exists():raw=dest.read_bytes();record['source']='existing_local'
                else:
                    url=f'https://api.opendota.com/api/matches/{mid}'+('?' + urlencode({'api_key':key}) if key else '')
                    try:
                        with urlopen(Request(url,headers={'User-Agent':'Dota2-reconstruction/1.0'}),timeout=60) as response:raw=response.read()
                    finally:time.sleep(a.interval)
                    validate(raw,mid)
                    with dest.open('xb') as f:f.write(raw)
                    record['source']='OpenDota_API'
                validate(raw,mid);record['observed_sha256']=sha(raw);record['status']='byte_identical' if sha(raw)==r['sha256'] else 'byte_hash_differs'
            except HTTPError as e:record.update(status='http_error',http_status=e.code)
            except Exception as e:record.update(status='error',error_type=type(e).__name__)
            # Never log request URLs, exception text or credentials.
            log.write(json.dumps(record)+'\n');log.flush();counts[record['status']]=counts.get(record['status'],0)+1
            if record.get('http_status') in (401,403,429):break
    print(json.dumps({'attempted':sum(counts.values()),'planned_this_call':len(rows),'statuses':counts}))
    if counts.get('error') or counts.get('http_error'):raise SystemExit(2)
if __name__=='__main__':main()
