"""
Unit and integration sanity tests for L2D-SDDA pipeline
"""

import unittest
import numpy as np
import pandas as pd
from src.config import DATASET_PATH
from src.data_loader import load_dataset, get_feature_columns, get_window_data
from src.deferral import compute_composite_score, apply_deferral_budget


class TestPipeline(unittest.TestCase):

    def test_data_loader(self):
        df = load_dataset()
        self.assertGreater(len(df), 0)
        self.assertIn("Churn", df.columns)
        self.assertIn("window_id", df.columns)

    def test_feature_columns(self):
        df = load_dataset()
        features = get_feature_columns(df)
        self.assertEqual(len(features), 32)
        self.assertNotIn("Churn", features)
        self.assertNotIn("window_id", features)

    def test_deferral_scoring(self):
        u = np.array([0.2, 0.8, 0.5])
        d = np.array([0.1, 0.4, 0.9])
        scores = compute_composite_score(u, d, lambda_param=0.15)
        self.assertEqual(len(scores), 3)
        expected_0 = 0.2 + 0.1 + 0.15 * (0.2 * 0.1)
        self.assertAlmostEqual(scores[0], expected_0, places=4)

    def test_deferral_budget(self):
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        # 40% budget -> top 2 deferred
        deferred = apply_deferral_budget(scores, budget=0.4)
        self.assertEqual(deferred.sum(), 2)
        self.assertEqual(deferred[1], 1)
        self.assertEqual(deferred[3], 1)


if __name__ == "__main__":
    unittest.main()
