from precis import stats as gs
import numpy as np
import numpy.testing as npt
import unittest


class TestBase(unittest.TestCase):

    def test_cronbach_alpha(self):
        scores = np.array([[ 3,4,3,3],
                  [ 5,4,4,3],
                  [ 1,3,4,5], 
                  [ 4,5,4,2],
                  [ 1,3,4,1],
                  [ 3,3,3,1],
                  [ 3,4,5,1],
                  [ 1,2,1,3]])
        # Test three scales: one is sum of all items; second has reverse keyed items;
        # third omits an item. Correct alphas from the psych package in R. Values for 
        # second and third examples are nonsensical, but that's okay for testing.
        alphas = gs.cronbach_alpha(scores[:,:4], np.array([
          [1,1,1,1],
          [1,-1,-1,1],
          [0,-1,1,0]]).T)
        npt.assert_array_almost_equal(alphas, np.array([0.409, -0.461, -3.333]))

if __name__ == '__main__':
    unittest.main()