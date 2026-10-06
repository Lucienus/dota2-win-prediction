"""Safety checks for the corrected and portable execution path."""
from pathlib import Path
import sys,unittest,tempfile,json
import copy
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from release_common import load_cache,sha
from data_protocol import load_manifest
from v4_objectives import tower_counts_before,TowerCorrectedExtractor
from models import DotaMultiModalPredictor,VARIANTS
from experiment import collate_prefixes

def synthetic_match():
    players=[{'player_slot':i if i<5 else 128+i-5,'hero_id':i+1,
              'times':list(range(0,1260,60)),'gold_t':[100+m*(i+10) for m in range(21)],
              'xp_t':[50+m*20 for m in range(21)],'lh_t':list(range(21)),'dn_t':[0]*21,
              'purchase_log':[{'time':550,'key':'blink'}],'kills_log':[]} for i in range(10)]
    return {'match_id':42,'start_time':1700000000,'duration':1201,'radiant_win':True,'players':players,
            'objectives':[{'time':600,'type':'building_kill','key':'npc_dota_badguys_tower1_mid'}]}

class Contracts(unittest.TestCase):
    def test_future_events_and_outcome_do_not_change_inputs(self):
        match=synthetic_match();before=TowerCorrectedExtractor(match,10).extract_inputs()
        changed=copy.deepcopy(match);changed['radiant_win']=False
        changed['objectives'].append({'time':601,'type':'building_kill','key':'npc_dota_badguys_tower2_mid'})
        for player in changed['players']:
            for key in ('gold_t','xp_t','lh_t','dn_t'):player[key][11:]=[999999]*10
            player['purchase_log'].append({'time':601,'key':'future_item'})
            player.update(actions_per_min=999999,lane_role=1,hero_damage=999999)
        after=TowerCorrectedExtractor(changed,10).extract_inputs()
        for key in before:np.testing.assert_array_equal(before[key],after[key])
    def test_finished_matches_rejected(self):
        match=synthetic_match();match['duration']=600
        with self.assertRaises(ValueError):TowerCorrectedExtractor(match,10)
    def test_all_neural_variants_backward(self):
        torch.set_num_threads(4)
        batch=collate_prefixes([{k:torch.as_tensor(v) for k,v in TowerCorrectedExtractor(synthetic_match(),10).extract().items()}])
        for name,config in VARIANTS.items():
            with self.subTest(variant=name):
                model=DotaMultiModalPredictor(**config);out=model(batch)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],batch['outcome'])
                if name!='single_task':loss=loss+.1*(out['aux']-batch['aux']).square().sum()
                loss.backward();self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_tower_cutoff_and_duplicate_type_keys(self):
        events=[{'type':'building_kill','key':'npc_dota_badguys_tower4','time':x} for x in (100,110,1000)]
        self.assertEqual(tower_counts_before({'objectives':events},200),(2,0))
        self.assertEqual(tower_counts_before({'objectives':events},99),(0,0))
    def test_missing_log_is_not_zero_towers(self):
        with self.assertRaises(ValueError):tower_counts_before({},200)
    def test_tampered_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'sample.bin';f.write_bytes(b'original')
            (p/'metadata.json').write_text(json.dumps({'files':{'sample.bin':sha(f)}}))
            load_cache(p);f.write_bytes(b'changed')
            with self.assertRaises(ValueError):load_cache(p)
    def test_frozen_split_has_no_duplicate_matches(self):
        if not (ROOT/'inputs/historical_manifest.json').exists():
            self.skipTest('Frozen study manifest must be obtained separately')
        m=load_manifest(ROOT/'inputs/historical_manifest.json')
        self.assertEqual([len(m['splits'][s]) for s in ('train','validation','test')],[34958,7491,7491])

if __name__=='__main__':unittest.main()
