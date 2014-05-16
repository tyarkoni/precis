import unittest
import pandas as pd
import numpy as np
from scythe.generate import Generator
from scythe.base import Measure
from scythe import abbreviate
from scythe import plot as sp
import matplotlib.pyplot as plt
from helpers import get_test_data_path as tdp
from os.path import join


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        measure = Measure(X=join(tdp(), 'items.txt'), y=join(tdp(), 'scales.txt'), key=join(tdp(), 'key.txt'), missing='drop')
        abb = abbreviate.TopNAbbreviator(max_items=8, min_r=0.3)
        gen = Generator(abbreviator=abb)
        am = gen.run(measure, n_gens=3)
        cls.generator = gen

    def test_scale_intercorrelation_plot(self):
        sp.scale_correlation_matrix(self.generator.best)
    
    def test_evolution_plot(self):
        sp.history(self.generator)

if __name__ == '__main__':
    unittest.main()