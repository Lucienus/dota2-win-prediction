import json,sys,unittest,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fetch_frozen_matches import rows_from,validate
class ReconstructionTests(unittest.TestCase):
    def test_duplicate_rejected(self):
        row={'match_id':1,'file':'1.json'}
        with self.assertRaises(ValueError):rows_from({'included':[row,row]})
    def test_path_rejected(self):
        with self.assertRaises(ValueError):rows_from({'included':[{'match_id':1,'file':'../1.json'}]})
    def test_wrong_response_rejected(self):
        with self.assertRaises(ValueError):validate(json.dumps({'match_id':2,'players':[{}]*10}).encode(),1)
    def test_no_missing_replacement_and_order(self):
        m=json.loads((ROOT/'manifests/later.json').read_text());rows=rows_from(m)
        self.assertEqual(len(rows),1199);self.assertEqual(len(m['excluded']),1)
        for minute,ids in m['minute_ids'].items():
            self.assertEqual(set(ids),{r['match_id'] for r in rows if int(minute) in r['eligible_minutes']})
            self.assertEqual(len(ids),len(set(ids)))
    def test_changed_raw_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw';raw.mkdir();out=Path(d)/'rebuilt'
            m=json.loads((ROOT/'manifests/later.json').read_text())
            for row in m['included']:(raw/row['file']).write_text(json.dumps({'match_id':row['match_id'],'players':[{}]*10}))
            p=subprocess.run([sys.executable,str(ROOT/'code/rebuild_frozen.py'),'--cohort','later','--raw-dir',str(raw),'--prior',str(Path(d)/'unused.json'),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('raw hashes differ',p.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
