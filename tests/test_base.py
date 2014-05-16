import unittest
from scythe.base import Dataset, Measure
from scythe.generate import Generator
import pandas as pd
import numpy as np
from helpers import get_test_data_path as tdp
from os.path import join, exists
import tempfile
import shutil

class TestBase(unittest.TestCase):

    def setUp(self):
        # pandas DFs for items and scales
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=['i1', 'i2', 'i3', 'i4', 'i5'])
        self.y = pd.DataFrame(np.random.rand(100, 2), columns=['s1', 's2'])
        self.key = pd.DataFrame(np.array([[0,1,1,0,1],[1,0,1,1,0]]).T)

    def test_dataset_init(self):
        # From text files
        dataset = Dataset(join(tdp(), 'items.txt'), join(tdp(), 'scales.txt'))
        self.assertEqual(len(dataset.X), len(dataset.y))
        # From pandas DFs
        dataset = Dataset(self.X, self.y)
        self.assertEqual(len(dataset.X), len(dataset.y))

    def test_dataset_process_missing_data(self):
        # Drop missing values
        X = self.X.copy()
        X.iloc[[2,7],2] = np.nan
        dataset = Dataset(X, self.y)
        dataset.process_missing_data('drop')
        self.assertEqual(len(dataset.X), 98)
        # Impute missing values
        mean2 = X.ix[:,2].mean()
        dataset = Dataset(X, self.y)
        dataset.process_missing_data('impute')
        self.assertEqual(dataset.X.ix[7,2], mean2)

    def test_dataset_select_subjects(self):
        dataset = Dataset(self.X, self.y)
        dataset.select_subjects(range(90))
        self.assertEqual(len(dataset.X), 90)

    def test_dataset_select_items(self):
        dataset = Dataset(self.X, self.y)
        # by column name
        dataset.select_X(['i1', 'i2'])
        self.assertEqual(list(dataset.X.columns), ['i1', 'i2'])
        # by index
        dataset = Dataset(self.X, self.y)
        dataset.select_X([0, 3])
        self.assertEqual(list(dataset.X.columns), ['i1', 'i4'])

    def test_dataset_reverse_items(self):
        dataset = Dataset(self.X, self.y)
        m0, m1 = dataset.X.ix[:,0].mean(), dataset.X.ix[:,1].mean()
        dataset.reverse_items([0, 2], max_score=5)
        self.assertAlmostEqual(dataset.X.ix[:,0].mean(), 6 - m0)
        self.assertAlmostEqual(dataset.X.ix[:,1].mean(), m1)

    def test_dataset_init_measure(self):
        dataset = Dataset(self.X, self.y)
        self.assertRaises(ValueError, Measure)
        measure = Measure(dataset=dataset, key=self.key)  # From Dataset

    def test_measure_trim(self):
        measure = Measure(X=self.X, y=self.y)
        key = np.array([[1,-1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]])
        self.assertEqual(key.shape, (4,4))
        measure.key = key
        measure.trim(data=False)
        self.assertEqual(measure.key.shape, (3,3))

    def test_measure_compute_stats(self):
        measure = Measure(X=self.X, y=self.y, key=self.key)
        measure.compute_stats()
        self.assertEqual(measure.alpha.shape, (2,))
        self.assertEqual(measure.alpha.dtype, 'float64')
        self.assertEqual(measure.r_squared.shape, (2,))
        self.assertEqual(measure.alpha.dtype, 'float64')
        self.assertTrue(~np.isnan(measure.r_squared).any())

    def test_measure_save(self):
        measure = Measure(X=join(tdp(), 'items.txt'), y=join(tdp(), 'scales.txt'), key=join(tdp(), 'key.txt'), missing='drop')
        t = tempfile.mkdtemp()
        measure.save(key=True, summary=True, pickle=True, path=t)
        self.assertTrue(exists(t + '/key.txt'))
        self.assertTrue(exists(t + '/data.pkl'))
        self.assertTrue(exists(t + '/summary.txt'))
        shutil.rmtree(t)

    def test_abbreviator(self):
        measure = Measure(X=join(tdp(), 'items.txt'), y=join(tdp(), 'scales.txt'), key=join(tdp(), 'key.txt'), missing='drop')
        gen = Generator()
        am = gen.run(measure, n_gens=3)


if __name__ == '__main__':
    unittest.main()
