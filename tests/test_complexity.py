"""Measurement accounting and table integrity, without fitting a model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from benchmark_complexity import measure, positive
from render_complexity import validate,tables,paper_reference


class ComplexityTests(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'benchmark_reference/measurements.json'
        if not self.path.exists():
            self.skipTest('Recorded benchmark results are excluded from the code-only release')
        self.data=json.loads(self.path.read_text('utf-8'))

    def test_reference_has_all_36_conditions(self):
        conditions=validate(self.data)
        self.assertEqual(len(conditions)*len(self.data['neural']),36)
        self.assertIn('Node MLP',tables(self.data)[0])

    def test_incomplete_run_is_rejected(self):
        self.data['status']='running'
        with self.assertRaisesRegex(ValueError,'Incomplete'):validate(self.data)

    def test_missing_or_duplicate_condition_is_rejected(self):
        row=self.data['neural'][0];row['timings'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):validate(self.data)
        row['timings'].append(copy.deepcopy(row['timings'][0]))
        with self.assertRaisesRegex(ValueError,'coverage'):validate(self.data)

    def test_fabricated_summary_is_rejected(self):
        self.data['neural'][0]['timings'][0]['median_ms']+=1
        with self.assertRaisesRegex(ValueError,'Summary'):validate(self.data)

    def test_sample_count_and_nonfinite_values_are_rejected(self):
        t=self.data['neural'][0]['timings'][0];sample=t['measurements_ms'].pop()
        with self.assertRaisesRegex(ValueError,'sample count'):validate(self.data)
        t['measurements_ms'].append(float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):validate(self.data)

    def test_parameter_accounting_is_checked(self):
        self.data['neural'][0]['parameters']+=1
        with self.assertRaisesRegex(ValueError,'component'):validate(self.data)

    def test_cpu_only_table_does_not_invent_gpu_cells(self):
        self.data['protocol']['devices']=['cpu']
        for r in self.data['neural']:r['timings']=[t for t in r['timings'] if t['device']=='cpu']
        tex,md=tables(self.data)
        self.assertIn('CPU, batch 64',tex);self.assertNotIn('CUDA, batch',tex)

    def test_paper_reference_requires_frozen_bytes(self):
        text=paper_reference(self.path,self.data)
        self.assertNotIn('@@',text);self.assertIn('1.268',text)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'changed.json';p.write_text(json.dumps(self.data))
            with self.assertRaisesRegex(ValueError,'frozen'):paper_reference(p,self.data)

    def test_timing_excludes_warmup_and_disables_gradients(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,batch):
                assert not self.training and not torch.is_grad_enabled()
                self.calls+=1;return batch['x']+1
        model=Model()
        result=measure(model,{'x':torch.zeros(2)},torch.device('cpu'),3,5)
        self.assertEqual(model.calls,8);self.assertEqual(len(result['measurements_ms']),5)
        self.assertTrue(all(t>0 for t in result['measurements_ms']))


if __name__=='__main__':unittest.main()
