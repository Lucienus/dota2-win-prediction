"""Regression tests for paper-only selection; no training or real data needed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from later_scope import PAPER_MODELS, PAPER_MINUTES, PAPER_SEEDS, select_runs


class LaterScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.runs.mkdir()

    def add(self, model='full', minute=10, seed=17, folder=None):
        p = self.runs / (folder or f'{model}_{minute}_{seed}')
        p.mkdir()
        (p / 'result.json').write_text(json.dumps(dict(model=model, minute=minute, seed=seed)))
        (p / 'model.pt').write_bytes(b'presence-only fixture; never loaded')
        return p

    def grid(self):
        for values in product(PAPER_MODELS, PAPER_MINUTES, PAPER_SEEDS):
            self.add(*values)

    def test_full_training_grid_selects_only_paper_84(self):
        self.grid()
        for values in product(('gru','no_role','complete_graph','no_draft_branch',
                               'mean_draft','single_task','random_forest'),PAPER_MINUTES,PAPER_SEEDS):
            self.add(*values)
        selected, audit = select_runs(self.runs)
        self.assertEqual(len(selected),84)
        self.assertEqual(audit['ignored_count'],84)
        self.assertIn('no_graph',{r['model'] for _,r,_ in selected})
        self.assertEqual({r['model'] for _,r,_ in selected},set(PAPER_MODELS))
        self.assertEqual(len(select_runs(self.runs,'all')[0]),168)

    def test_missing_combination_fails_even_if_an_extra_replaces_it(self):
        self.grid()
        (self.runs/'full_10_17'/'result.json').unlink()
        self.add('random_forest')
        with self.assertRaisesRegex(ValueError,'found 83/84'):
            select_runs(self.runs)

    def test_duplicate_identity_fails(self):
        self.grid();self.add(folder='duplicate')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            select_runs(self.runs)

    def test_unexpected_seed_fails(self):
        self.grid();self.add(seed=99)
        with self.assertRaisesRegex(ValueError,'Unexpected paper time/seed'):
            select_runs(self.runs)

    def test_missing_or_ambiguous_checkpoint_fails(self):
        p=self.add();(p/'model.pt').unlink()
        with self.assertRaisesRegex(ValueError,'exactly one checkpoint'):
            select_runs(self.runs,'all')
        (p/'model.pt').touch();(p/'model.joblib').touch()
        with self.assertRaisesRegex(ValueError,'exactly one checkpoint'):
            select_runs(self.runs,'all')

    def test_partial_requires_explicit_all(self):
        self.add()
        with self.assertRaisesRegex(ValueError,'Incomplete paper grid'):
            select_runs(self.runs)
        selected,audit=select_runs(self.runs,'all')
        self.assertEqual(len(selected),1)
        self.assertFalse(audit['paper_grid_complete'])

    def test_empty_grid_fails(self):
        for scope in ('paper','all'):
            with self.assertRaises(ValueError):select_runs(self.runs,scope)

    def test_cli_default_dry_run_and_failure_write_nothing(self):
        self.grid();out=self.root/'output'
        command=[sys.executable,str(ROOT/'code/evaluate_later.py'),'--runs',str(self.runs),
                 '--later',str(self.root/'no_data_needed_for_selection'),
                 '--output',str(out),'--dry-run']
        proc=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        audit=json.loads(proc.stdout)
        self.assertEqual(audit['scope'],'paper');self.assertEqual(audit['selected_count'],84)
        self.assertFalse(out.exists())
        (self.runs/'full_10_17'/'result.json').unlink()
        proc=subprocess.run(command[:-1],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('Incomplete paper grid',proc.stderr)
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
