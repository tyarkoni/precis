import unittest
import pandas as pd
import numpy as np
from scythe import evaluate, abbreviate
from scythe.base import Dataset, Measure
from helpers import get_test_data_path as tdp
from os.path import join

class TestBase(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset(join(tdp(), 'items.txt'), join(tdp(), 'scales.txt'), missing='drop')

    def test_abbreviate_get_X_y(self):
        abb = abbreviate.TopNAbbreviator()
        # Without any selection
        abb.select = None
        X, y = abb.get_X_y(self.dataset)
        self.assertEqual(X.shape, self.dataset.X.shape)
        self.assertEqual(y.shape, self.dataset.y.shape)
        # With selection
        abb.select = [i for i in range(X.shape[1]) if i % 2 == 0]
        X, y = abb.get_X_y(self.dataset)
        self.assertEqual(X.shape[1], self.dataset.X.shape[1]/2)
        self.assertEqual(y.shape, self.dataset.y.shape)

    def test_top_n_abbreviator(self):
        abb = abbreviate.TopNAbbreviator(max_items=5, min_r=0.2)
        abb.abbreviate_apply(self.dataset)
        key = abb.key
        self.assertEqual(key.shape, (154,8))
        self.assertEqual(np.sum(key.any(axis=1)), 40)

    def test_yarkoni_evaluator(self):
        # loss = evaluate.YarkoniEvaluator()
        # key = loss.make_key(self.dataset.X.values, self.dataset.y.values)
        # n_items_used = np.sum(key.any(axis=1))
        # self.assertEqual(n_items_used, 40)
        # measure = Measure(self.dataset, key=key)
        # measure.compute_stats()
        # self.assertTrue(np.all(measure.r_squared > 0.2))
        pass

if __name__ == '__main__':
    unittest.main()