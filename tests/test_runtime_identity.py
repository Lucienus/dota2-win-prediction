"""Resume guards and a small real CPU interruption/resumption regression."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from runtime_identity import collect_environment, dependency_versions, DEPENDENCIES
from experiment import open_run,validate_progress
from data_protocol import write_json,fingerprint
import train_neural
import train_trees
import train_linear_boosting


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'run'
        self.spec={'environment':{'schema_version':2,'dependencies':{n:'1.0' for n in DEPENDENCIES}},
                   'cache_id':'data-a','source_hashes':{'train.py':'abc'},'seed':17,'epochs':30}

    def test_actual_imported_versions_and_stable_snapshot(self):
        import scipy,joblib,threadpoolctl,xgboost
        versions=dependency_versions()
        for name,module in [('scipy',scipy),('joblib',joblib),('threadpoolctl',threadpoolctl),('xgboost',xgboost)]:
            self.assertEqual(versions[name],module.__version__)
        self.assertEqual(set(versions),set(DEPENDENCIES))
        a=collect_environment();b=collect_environment()
        self.assertEqual(a,b)
        self.assertNotIn('filepath',json.dumps(a))

    def test_missing_optional_dependency_is_explicit_and_broken_import_fails(self):
        import importlib
        real=importlib.import_module
        def missing(name):
            if name=='lightgbm':raise ModuleNotFoundError('missing',name=name)
            return real(name)
        with patch('runtime_identity.importlib.import_module',side_effect=missing):
            self.assertIsNone(dependency_versions()['lightgbm'])
        def broken(name):
            if name=='lightgbm':raise ModuleNotFoundError('broken',name='internal_dependency')
            return real(name)
        with patch('runtime_identity.importlib.import_module',side_effect=broken):
            with self.assertRaisesRegex(RuntimeError,'Broken dependency'):dependency_versions()

    def test_same_run_skips_complete_result_without_mutation(self):
        self.assertIsNone(open_run(self.folder,self.spec,resume=True))
        write_json(self.folder/'result.json',{'specification':self.spec,'test':{'accuracy':.5}})
        before={p.name:p.read_bytes() for p in self.folder.iterdir()}
        result=open_run(self.folder,self.spec,resume=True)
        self.assertEqual(result['test']['accuracy'],.5)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.folder.iterdir()})

    def test_every_dependency_change_rejects_without_writes(self):
        open_run(self.folder,self.spec,resume=True)
        before=(self.folder/'run_config.json').read_bytes()
        for dep in DEPENDENCIES:
            with self.subTest(dependency=dep):
                changed=copy.deepcopy(self.spec);changed['environment']['dependencies'][dep]='2.0'
                with self.assertRaisesRegex(ValueError,'environment.dependencies.'+dep):
                    open_run(self.folder,changed,resume=True)
                self.assertEqual(before,(self.folder/'run_config.json').read_bytes())

    def test_data_code_parameters_and_hardware_changes_reject(self):
        open_run(self.folder,self.spec,resume=True)
        for key,value in [('cache_id','data-b'),('source_hashes',{'train.py':'xyz'}),('epochs',31),('seed',29)]:
            changed=copy.deepcopy(self.spec);changed[key]=value
            with self.assertRaisesRegex(ValueError,key):open_run(self.folder,changed,resume=True)
        changed=copy.deepcopy(self.spec);changed['environment']['cuda_devices']=[{'name':'new GPU'}]
        with self.assertRaisesRegex(ValueError,'cuda_devices'):open_run(self.folder,changed,resume=True)

    def test_legacy_or_inconsistent_completed_result_rejects(self):
        self.folder.mkdir();write_json(self.folder/'run_config.json',{'environment':{'python':'old'}})
        with self.assertRaisesRegex(ValueError,'dependencies'):open_run(self.folder,self.spec,resume=True)
        write_json(self.folder/'run_config.json',self.spec)
        write_json(self.folder/'result.json',{'specification':{'seed':29}})
        with self.assertRaisesRegex(ValueError,'result.json'):open_run(self.folder,self.spec,resume=True)

    def test_progress_is_bound_to_configuration(self):
        validate_progress({'specification_sha256':fingerprint(self.spec)},self.spec)
        for progress in [{},{'specification_sha256':'another-run'}]:
            with self.assertRaisesRegex(ValueError,'different or legacy'):validate_progress(progress,self.spec)


class TrainerGuardIntegration(unittest.TestCase):
    def check_direct_trainer(self,module,model):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);meta={'cache_id':'fixture','files':{}}
            env={'schema_version':2,'dependencies':{'scipy':'1.0'}}
            with patch.object(module,'RESULTS',root),patch.object(module,'CACHE',root),patch.object(module,'environment',return_value=env):
                with patch.object(module,'PrefixDataset',side_effect=InterruptedError('stop before data')):
                    with self.assertRaises(InterruptedError):module.train(meta,10,model,17)
                folder=root/f'{model}_10m_seed17';spec=json.loads((folder/'run_config.json').read_text())
                write_json(folder/'result.json',{'specification':spec})
                with patch.object(module,'PrefixDataset',side_effect=AssertionError('must skip before data')):
                    module.train(meta,10,model,17)
                    env['dependencies']['scipy']='2.0'
                    with self.assertRaisesRegex(ValueError,'environment.dependencies.scipy'):
                        module.train(meta,10,model,17)

    def test_neural_entry_checks_before_loading_data(self):
        self.check_direct_trainer(train_neural,'full')

    def test_tree_entry_checks_before_loading_data(self):
        self.check_direct_trainer(train_trees,'random_forest')

    def test_linear_entry_checks_before_candidate_fit(self):
        m=train_linear_boosting
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);env={'schema_version':2,'dependencies':{'lightgbm':'1.0'}}
            argv=['train','--cache',str(root),'--output',str(root),'--models','gold_logistic','--minutes','10','--seeds','17']
            with patch.object(sys,'argv',argv),patch.object(m,'load_cache',return_value={'cache_id':'fixture','files':{}}),patch.object(m,'PrefixDataset'),patch.object(m,'baseline_features'),patch.object(m,'environment',return_value=env),patch.object(m,'candidates',side_effect=InterruptedError('stop before fitting')):
                with self.assertRaises(InterruptedError):m.main()
                folder=root/'gold_logistic_10m_seed17';spec=json.loads((folder/'run_config.json').read_text())
                write_json(folder/'result.json',{'specification':spec})
                m.main()
                env['dependencies']['lightgbm']='2.0'
                with self.assertRaisesRegex(ValueError,'environment.dependencies.lightgbm'):m.main()


class NeuralResumeIntegration(unittest.TestCase):
    def test_interrupted_cpu_subset_matches_uninterrupted(self):
        if not all((ROOT/f'inputs/historical/{split}_10m.npz').exists() for split in ('train','validation','test')):
            self.skipTest('Study feature arrays must be obtained separately')
        # An eight-match-per-split engineering fixture, not a paper retraining.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';cache.mkdir()
            for split in ('train','validation','test'):
                with np.load(ROOT/f'inputs/historical/{split}_10m.npz') as data:
                    y=data['outcome'];idx=np.concatenate([np.flatnonzero(y==0)[:4],np.flatnonzero(y==1)[:4]])
                    np.savez_compressed(cache/f'{split}_10m.npz',**{k:data[k][idx] for k in data.files})
            write_json(cache/'purchase_prior.json',{})
            meta={'cache_id':'eight-per-split-engineering-fixture','files':{}}
            original_save=train_neural.atomic_torch_save
            def interrupt(value,path):
                original_save(value,path)
                if path.name=='progress.pt' and value['epoch']==1:
                    raise InterruptedError('simulated after committed epoch one')
            with patch.object(train_neural,'CACHE',cache),patch('torch.cuda.is_available',return_value=False):
                with patch.object(train_neural,'RESULTS',root/'continuous'):
                    continuous=train_neural.train(meta,10,'full',17)
                with patch.object(train_neural,'RESULTS',root/'resumed'):
                    with patch.object(train_neural,'atomic_torch_save',side_effect=interrupt):
                        with self.assertRaises(InterruptedError):train_neural.train(meta,10,'full',17)
                    resumed=train_neural.train(meta,10,'full',17)
                    folder=root/'resumed/full_10m_seed17'
                    unchanged=(folder/'result.json').read_bytes()
                    again=train_neural.train(meta,10,'full',17)
                    self.assertEqual(unchanged,(folder/'result.json').read_bytes())
                    self.assertEqual(again,resumed)
            for name in ['predictions.npz','validation_predictions.npz']:
                with np.load(root/'continuous/full_10m_seed17'/name) as a,np.load(root/'resumed/full_10m_seed17'/name) as b:
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(continuous['history'],resumed['history'])
            self.assertEqual(continuous['best_epoch'],resumed['best_epoch'])


if __name__=='__main__':unittest.main()
