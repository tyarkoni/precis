""" Contains classes for evaluating abbreviated measures. """

import numpy as np
import abc


class Evaluator(object):

    ''' Base Evaluator class. '''

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, measure, individual):
        """ Apply the loss function to the AbbreviatedMeasure produced by 
        selecting the passed items from the passed measure.
        Args:
            measure: a Measure instance to abbreviate and evaluate.
            individual: the individual/chromosome to evaluate.
        """


class YarkoniEvaluator(Evaluator):

    def __init__(self, item_cost=0.05):
        """
        Args:
            item_cost: The increase in cost/loss associated with the addition 
                of each additional item. For instance, if item_cost = 0.1, a 
                measure with 100 items will have an associated total item cost 
                of 10. For details, see Yarkoni (2010).
        """
        self.item_cost = item_cost

    def evaluate(self, measure, weights=None):
        """ The loss function used in Yarkoni (2010). Basically, total loss is 
        just the sum of two components: (a) an item cost that increases in 
        direct proportion to the number of items retained, and (b) a variance 
        cost that increases in direct proportion to the amount of unexplained 
        variance in the original scales.
        """
        # Compute R^2
        d = measure.dataset
        pred_y = np.dot(d.X, measure.key)
        r_squared = (np.corrcoef(d.y, pred_y, rowvar=0)[
                     0:measure.n_y, measure.n_y::] ** 2).diagonal()

        # Item cost: just the scaled number of items kept
        item_cost = d.X.shape[1] * self.item_cost

        # Compute variance cost--just mean variance unaccounted for in each
        # scale
        if weights is not None:
            r_squared *= weights
        variance_cost = measure.n_y - np.sum(r_squared)

        return float(item_cost + variance_cost)
