import numpy as np
import abc
from precis.base import Measure


class Abbreviator(object):
    """ Base Abbreviator class."""
    __metaclass__ = abc.ABCMeta

    def get_X_y(self, data):
        """ Slices X data in Measure based on items, and returns X and y
        numpy arrays. """
        if isinstance(data, Measure):
            data = data.dataset

        X = data.X
        # print X.shape
        # print self.select
        if self.select is not None:
            X = X.iloc[:, self.select]
        y = data.y
        return (X.values, y.values)

    def abbreviate(self, data, select=None):
        """ Take input data and creates a new abbreviated scoring key.
        Args:
            data (Measure or Dataset): an instance of class Measure or
                Dataset.
            select (list): optional columns in X to extract before
                generating key.
        Returns:
            A scoring key represented as a 2D numpy array, with X in rows
            and y in columns.
        """
        if select is not None:
            select = np.where(select)[0]
        self.select = select
        X, y = self.get_X_y(data)
        self.key = self._make_key(X, y)

    def apply(self, data):
        X, y = self.get_X_y(data)
        return Measure(X=X, y=y, key=self.key)

    def abbreviate_apply(self, data, select=None):
        self.abbreviate(data, select)
        return self.apply(data)

    @abc.abstractmethod
    def _make_key(self, X, y):
        """ Scoring key generation method; must be overriden by subclasses. """


class TopNAbbreviator(Abbreviator):

    """
    A simple abbreviator that scores each scale using the n items that display
    the strongest absolute correlation with the scale score.

    Args:
        max_items (int): Maximum number of items that can be used to score a
            scale.
        min_r (float): Minimum absolute correlation an item must have with the
            full scale in order to be included in scoring.
    """

    def __init__(self, max_items=5, min_r=0.0):
        self.max_items = max_items
        self.min_r = min_r

    def _make_key(self, X, y):
        """ Scales are abbreviated as follows. First, for each scale, we
        rank-order all items by size of absolute correlation with total scale
        score. Next, we mask out all items with correlations less than min_r.
        Then, we retain only the top N items. The resulting (signed) set of
        items constitutes the scoring key for that scale. 
        """
        n_X, n_y = X.shape[1], y.shape[1]
        key = np.zeros((n_X, n_y))
        cors = np.corrcoef(X, y, rowvar=0)[0:n_X, n_X::]
        abs_cors = np.abs(cors)
        ranks = (-abs_cors).argsort(axis=0).argsort(axis=0) + 1
        ranks[abs_cors < self.min_r] = 0  # Drop correlations below threshold
        ranks[ranks > self.max_items] = 0  # Keep only top items
        return np.multiply(ranks.astype(bool), np.sign(cors))
