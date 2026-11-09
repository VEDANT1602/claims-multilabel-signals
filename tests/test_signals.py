import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core import FEATURES,LABELS,choose_thresholds,decisions,probabilities,binary_f1
from generate import generate
from support import verify_manifest

class SignalTests(unittest.TestCase):
    def test_known_f1(self):np.testing.assert_allclose(binary_f1([1,1,0],[1,0,1]),.5)
    def test_rare_label_withheld(self):
        y=np.zeros((100,4));y[:30,:3]=1;y[:2,3]=1;p=.2+.6*y
        threshold,audit=choose_thresholds(y,p);self.assertIsNone(threshold[3]);self.assertEqual('withheld',audit[3]['status'])
        self.assertTrue((decisions(p,threshold)[:,3]==-1).all())
    def test_threshold_selected_on_known_validation_answer(self):
        y=np.tile([[1,1,1,1],[0,0,0,0]],(50,1));p=np.where(y,.3,.1)
        threshold,_=choose_thresholds(y,p);self.assertTrue(all(.1<t<=.3 for t in threshold));self.assertTrue((decisions(p,threshold)==y).all())
    def test_threshold_nan_bool_out_of_range_rejected(self):
        for x in [float('nan'),True,1.1,-.1]:
            with self.assertRaises(ValueError):decisions(np.ones((3,4))*.5,[x]*4)
    def test_out_of_range_probabilities_refused(self):
        for x in [-.1,1.1]:
            with self.assertRaises(ValueError):decisions(np.full((3,4),x),[.5]*4)
    def test_invalid_feature_matrix_rejected(self):
        a=json.loads((ROOT/'models/linear-heads.json').read_text())
        for x in [np.ones((2,7)),np.full((2,8),float('nan'))]:
            with self.assertRaises(ValueError):probabilities(x,a)
    def test_portable_artifact_known_sigmoid(self):
        a={'features':FEATURES,'labels':LABELS,'mean':[0]*8,'scale':[1]*8,'coef':[[0]*8]*4,'intercept':[0]*4}
        np.testing.assert_allclose(probabilities(np.ones((3,8)),a),.5)
    def test_artifact_feature_order_not_silent(self):
        a=json.loads((ROOT/'models/linear-heads.json').read_text());a['features']=a['features'][::-1]
        with self.assertRaises(ValueError):probabilities(np.zeros((2,8)),a)
    def test_families_disjoint(self):
        rows=generate();g={s:{r['family'] for r in rows if r['split']==s} for s in ['train','validation','test']}
        self.assertFalse(g['train']&g['test'] or g['validation']&g['test'] or g['train']&g['validation'])
    def test_selection_contract_and_metric_population(self):
        s=json.loads((ROOT/'results/threshold-selection.json').read_text());m=json.loads((ROOT/'results/metrics.json').read_text())
        self.assertTrue(s['validation_only']);self.assertEqual(3,len(m['supported_decision_labels']));self.assertEqual(800,m['split_records']['test'])
    def test_committed_manifest(self):verify_manifest()
