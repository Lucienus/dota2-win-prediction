"""Guard against incorrect paper means, pairing and input provenance."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from aggregate_results import collect,summarize,validate_predictions,check_metrics,main,write,expected_grid
from aggregate_statistics import interval,loss
from aggregate_tables import render_tables
from evaluation import probability_metrics
from release_common import sha

class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.inputs=self.root/'inputs';self.runs=self.root/'runs';self.later=self.root/'later'
        for p in (self.inputs/'historical',self.inputs/'later',self.runs,self.later):p.mkdir(parents=True)
        self.ids=np.arange(4);self.y=np.array([0,1,0,1],dtype=np.float32);self.p=np.array([.1,.7,.4,.8])
        self.metrics=probability_metrics(self.y,self.p)
        for t in (10,20,30,40):
            for cohort in ('historical','later'):
                for split in (('train','validation','test') if cohort=='historical' else ('',)):
                    np.savez(self.inputs/cohort/(f'{split}_{t}m.npz' if split else f'{t}m.npz'),match_id=self.ids,outcome=self.y,patch=np.full(4,59),start_time=np.full(4,1777616200),sequence=np.zeros((4,t+1,4)))
        files={p.name:sha(p) for p in (self.inputs/'historical').glob('*.npz')}
        write(self.inputs/'historical/metadata.json',{'files':files})
        write(self.inputs/'later/metadata.json',{'feature_files':{p.name:sha(p) for p in (self.inputs/'later').glob('*.npz')}})
        self.spec={'model':'full','minute':30,'seed':17,'cache_files':files,'environment':{'python':'test-only'},'budget':30}
        self.folder=self.runs/'full_30m_seed17';self.folder.mkdir()
        write(self.folder/'run_config.json',self.spec)
        write(self.folder/'result.json',{'model':'full','minute':30,'seed':17,'specification':self.spec,'test':self.metrics,'parameters':100524})
        np.savez(self.folder/'predictions.npz',match_id=self.ids,labels=self.y,probabilities=self.p)
        (self.folder/'model.pt').write_bytes(b'never deserialized in aggregation')
        f=self.later/self.folder.name;f.mkdir()
        np.savez(f/'predictions.npz',match_id=self.ids,labels=self.y,probabilities=self.p)
        write(f/'result.json',{'model':'full','minute':30,'seed':17,'evaluation_scope':'all','checkpoint_sha256':sha(self.folder/'model.pt'),'metrics':self.metrics})
        write(self.later/'evaluation_scope.json',{'scope':'all','selected_count':1,'selected':[{'model':'full','minute':30,'seed':17}]})
    def collect(self):return collect(self.runs,self.later,self.inputs)
    def edit(self,p,change):
        r=json.loads(p.read_text());change(r);write(p,r)
    def test_real_metrics_and_single_seed(self):
        records,*_=self.collect();rows=summarize(records)
        self.assertEqual(len(records),2);self.assertEqual(len(expected_grid()),252)
        self.assertIsNone(rows[0]['accuracy_sd']);self.assertFalse(rows[0]['complete'])
    def test_sample_sd(self):
        records,*_=self.collect();r=records[0]
        rows=summarize([r|{'seed':s,'accuracy':v} for s,v in zip((17,29,43),(.5,.75,1.))])
        self.assertEqual(rows[0]['accuracy_mean'],.75);self.assertEqual(rows[0]['accuracy_sd'],.25)
        self.assertTrue(rows[0]['complete'])
    def test_duplicate_identity(self):
        import shutil
        shutil.copytree(self.folder,self.runs/'duplicate')
        with self.assertRaisesRegex(ValueError,'Duplicate configuration'):self.collect()
    def test_reordered_ids(self):
        np.savez(self.folder/'predictions.npz',match_id=self.ids[::-1],labels=self.y,probabilities=self.p)
        with self.assertRaisesRegex(ValueError,'IDs/order/labels'):self.collect()
    def test_duplicate_matches(self):
        np.savez(self.folder/'predictions.npz',match_id=[0,1,2,2],labels=self.y,probabilities=self.p)
        with self.assertRaisesRegex(ValueError,'Duplicate match'):self.collect()
    def test_invalid_probabilities(self):
        for probs in ([.1,.7,np.nan,.8],[.1,.7,1.1,.8]):
            np.savez(self.folder/'predictions.npz',match_id=self.ids,labels=self.y,probabilities=probs)
            with self.assertRaises(ValueError):self.collect()
    def test_stale_saved_metric(self):
        self.edit(self.folder/'result.json',lambda r:r['test'].update(accuracy=.1))
        with self.assertRaisesRegex(ValueError,'Saved metric'):self.collect()
    def test_mismatched_specification(self):
        self.edit(self.folder/'run_config.json',lambda r:r.update(budget=99))
        with self.assertRaisesRegex(ValueError,'specification mismatch'):self.collect()
    def test_mixed_seed_budget(self):
        import shutil
        f=self.runs/'full_30m_seed29';shutil.copytree(self.folder,f)
        self.edit(f/'run_config.json',lambda r:r.update(seed=29,budget=99))
        spec=json.loads((f/'run_config.json').read_text())
        self.edit(f/'result.json',lambda r:r.update(seed=29,specification=spec))
        with self.assertRaisesRegex(ValueError,'Mixed specifications'):self.collect()
    def test_wrong_checkpoint(self):
        self.edit(self.later/self.folder.name/'result.json',lambda r:r.update(checkpoint_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'different/missing training checkpoint'):self.collect()
    def test_missing_later_output(self):
        (self.later/self.folder.name/'result.json').unlink()
        with self.assertRaisesRegex(ValueError,'Later results incomplete'):self.collect()
    def test_modified_features(self):
        p=self.inputs/'historical/test_30m.npz';p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Input hash mismatch'):self.collect()
    def test_strict_partial_writes_nothing(self):
        out=self.root/'report'
        with self.assertRaisesRegex(ValueError,'Incomplete paper grid'):
            main(['--runs',str(self.runs),'--later-results',str(self.later),'--inputs',str(self.inputs),'--output',str(out)])
        self.assertFalse(out.exists())
    def test_existing_output_rejected(self):
        with self.assertRaisesRegex(ValueError,'new directory'):main(['--runs',str(self.runs),'--output',str(self.runs)])
    def test_partial_tables_have_no_paper_mean(self):
        records,pred,features,params,*_=self.collect();out=self.root/'table';out.mkdir()
        contrasts={k:{} for k in ('full_minus_xgboost','neural_alternative_minus_full','later_full_minus_comparator','baseline_extension')}
        render_tables(out,summarize(records),features,params,pred,contrasts,self.inputs)
        text=(out/'accuracy.tex').read_text()
        self.assertIn('Full model & -- & -- & -- & --',text)
    def test_loss_average_is_not_probability_ensemble(self):
        y=np.array([1,0]);a=np.array([.1,.9]);b=np.array([.9,.1])
        self.assertFalse(np.allclose((loss(y,a)+loss(y,b))/2,loss(y,(a+b)/2)))
    def test_constant_paired_difference(self):
        r=interval(np.full(4,.25),groups=[1,1,2,2],repeats=20)
        self.assertEqual(r['paired_match_ci95'],[.25,.25]);self.assertEqual(r['league_cluster_ci95'],[.25,.25])
    def test_undefined_metric_not_coerced_to_zero(self):
        r=probability_metrics(np.zeros(4),self.p)
        self.assertIsNone(r['roc_auc']);check_metrics(r,r)

if __name__=='__main__':unittest.main()
