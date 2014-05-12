
import numpy as np
import pandas as pd
import logging
from scythe import stats
import random
import copy
import os


logger = logging.getLogger('scythe')


class Dataset(object):

    """ Represents X/y data for Measure training and application, plus some 
    additional helper methods. Mostly just wraps pandas. """

    def __init__(self, X, y, sep='\t', missing=None, select_X=None, select_y=None):
        """
        Args:
            X: The item data. Either the name of a text file containing item scores,
                or a pandas DataFrame.
            y: The scale scores. Either the name of a text file containing 
                scale scores, or a pandas DataFrame or Series.
            sep: field separator in data files.
            missing: How to handle subjects with missing item scores.
                See process_missing_data() for options.
            select_X: An optional list of columns to keep in X (discards the rest).
                Useful when computing measure stats for a previously abbreviated measures.
            select_y: Optional list of columns to keep in y.
        """
        # Read in data
        if isinstance(X, basestring):
            X = pd.read_csv(X, sep=sep).convert_objects(convert_numeric=True)
            try:
                X = X.drop('sample', axis=1)
            except:
                pass

        if isinstance(y, basestring):
            y = pd.read_csv(y, sep=sep).convert_objects(convert_numeric=True)

        self.X = X
        self.y = y

        if hasattr(y, 'columns'):
            self.y_names = self.y.columns
        else:
            self.y_names = range(y.shape[1])

        if select_X is not None:
            self.select_X(select_X)

        if select_y is not None:
            self.select_y(select_y)

        if missing is not None:
            self.process_missing_data(missing)

        # Basic validation:
        # if items and scales have different N's, look for an ID column and keep intersection
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Number of subjects in item and scale matrices do not match!")

        self.n_subjects = len(self.X)

        # Store item and scale counts
        self.n_X = self.X.shape[1]
        self.n_y = self.y.shape[1]


    def process_missing_data(self, missing='drop'):
        """ Process rows in item array that contain missing values.
        Args:
            missing (string): Method for dealing with missing values. Options:
                'drop': Drop any subjects with at least one missing items
                'impute': Impute the mean for that item across all subjects
        """
        if missing == 'drop':
            inds = pd.notnull(self.X).all(1).nonzero()[0]
            inds = np.intersect1d(inds, pd.notnull(self.y).all(1).nonzero()[0])
            n_missing = len(self.X) - len(inds)

            if n_missing:
                # Slice and reindex X and y
                self.X = self.X.ix[inds]
                self.y = self.y.ix[inds]
                logger.info('Found and deleted %d subjects with missing data.' % n_missing)

        # Imputation. Note that we don't impute the y values, because these should really be 
        # inspected and validated by the user before abbreviating.
        elif missing == 'impute':
            self.X = self.X.apply(lambda x: x.fillna(x.mean()), axis=0)
            # self.y = self.y.apply(lambda x: x.fillna(x.mean()), axis=0)


    def select_subjects(self, inds):
        ''' Trims X and y data to a subset of subjects. '''
        self.X = self.X.iloc[inds,:]
        self.y = self.y.iloc[inds,:]
        self.n_subjects = len(self.X)


    def select_X(self, cols):
        ''' Trims X to only the specified items. '''
        self.X = self.X.ix[:,cols]
        self.n_X = self.X.shape[1]


    def select_y(self, cols):
        ''' Trims X to only the specified items. '''
        self.y = self.y.ix[:,cols]
        self.n_y = self.y.shape[1]


    def score(self, key):
        y = np.dot(self.X, key)
        self.y = pd.DataFrame(y, columns=self.y.columns)


    def trim(self, subjects=None, items=None, key_only=True):
        ''' Trim data based on specific subjects, items, or the current key. 
        Args:
            subjects: optional list of subjects to keep.
            items: optional list of items to keep.
            key_only: if True, removes all X and y columns that aren't used 
                at least once in the scoring key. Also update the key.
        '''

        if key_only:
            x_inds = self.key.any(axis=1)
            y_inds = self.key.any(axis=0)


        if subjects is not None:
            self.select_subjects(subjects)

        if items is not None:
            self.select_X(items)


    def reverse_items(self, items, max_score=None):
        ''' Reverse scores on the items in the list. 
        Args:
            items: A list of item numbers to reverse. Items should be indexed from 1 and 
                not 0--i.e., pass in the number of the item on the scale.
            max_score: The value of the highest anchor (e.g., on a 5-point likert, 5.)
                If no value is passed, use the single highest value across the whole matrix.
        '''
        if max_score is None:
            max_score = np.max(self.X)
        self.X.ix[:, items] = max_score - self.X.ix[:, items] + 1 



class Measure(object):

    def __init__(self, dataset=None, X=None, y=None, key=None, trim=False, **kwargs):
        ''' Initialize a new measure. One of dataset or X must be passed.
        Args:
            dataset: Optional dataset to initialize with.
            X: Optional item data to pass to Dataset initializer.
            y: Optional scale scores to pass to Dataset initializer.
            key: An optional text file providing the scoring key (items x scales).
            trim: When True, drops all X/y columns not used in scoring key.
            kwargs: Additional keyword arguments to pass on to the Dataset initializer.
        '''

        if dataset is None:
            if X is None:
                raise ValueError("Either a Dataset or an X matrix must be provided.")
            dataset = Dataset(X, y, **kwargs)
        self.dataset = dataset

        if key is not None:
            if  isinstance(key, basestring):
                key = pd.read_csv(key, sep='\t', header=None)
            if isinstance(key, pd.DataFrame):
                key = key.values

        self.key = key

        if trim:
            self.trim()


    def trim(self, key=True, data=True):
        ''' Keep only X and y columns that are non-zero in the key.
        Args:
            key: If True, eliminates all-zero rows/cols from key.
            data: If True, applies trimming to X/y data in Dataset.

        '''
        if not hasattr(self, 'key'):
            raise AttributeError("No key found in current Measure, so trimming is not possible.")

        X_keep = np.any(self.key, axis=1)
        y_keep = np.any(self.key, axis=0)

        if key:
            self.key = self.key[X_keep]
            self.key = self.key[:,y_keep]
        if data:
            self.dataset.select_X(X_keep)
            self.dataset.select_y(y_keep)


    def score(self):
        ''' Compute y scores from X data and scoring key. Note: will overwrite any
        existing y data. '''
        self.dataset.score(self.key)


    def compute_stats(self):
        ''' Compute several statistics and metrics. '''

        dataset = self.dataset

        # Inter-scale correlation matrix
        self.y_corrs = np.corrcoef(dataset.y, rowvar=0)

        if self.key is not None:
            # Cronbach's alpha
            self.alpha = stats.cronbach_alpha(dataset.X.values, self.key)

            # Predicted scores
            self.predicted_y = np.dot(dataset.X, self.key)

            # R-squared
            self.r_squared = (np.corrcoef(dataset.y, self.predicted_y, rowvar=0)[0:self.n_y, self.n_y::] ** 2).diagonal()

            # Number of items per scale
            self.n_items_per_scale = np.sum(np.abs(self.key), 0)

            # Correlation matrix for predicted scores
            self.predicted_y_corrs = np.corrcoef(self.predicted_y, rowvar=0)


    def __str__(self):
        ''' Represent measure as a string. '''

        if not hasattr(self, 'predicted_y'):
            self.compute_stats()

        output = []
        output.append('Number of items: %d' % self.n_X)
        output.append('Number of scales: %d' % self.n_y)
        # output.append('Items used from original scale: %s' % ', '.join(str(x+1) for x in self.original_items))

        # Human-readable scoring key
        if self.key is not None:
            output.append('\nScoring key:')
            names = self.dataset.y_names
            item_labs = range(len(self.key))

            for s in range(self.n_y):
                item_list = []
                items_used = np.where(self.key[:,s] != 0)[0]
                for i, v in enumerate(items_used):
                    item = str(item_labs[v] + 1)
                    if self.key[v,s] < 0: item += 'R'
                    # item = '%dR' % (old_num+1) if self.key[i,s] < 0 else str(old_num+1)
                    item_list.append(item)
                output.append('%s (%d items, R^2=%.2f, alpha=%.2f):\t%s' % 
                    (names[s], self.n_items_per_scale[s], self.r_squared[s], 
                        self.alpha[s], ', '.join(item_list)))

        return '\n'.join(output)


    def __repr__(self):
        return self.__str__()


    def __getattr__(self, attr):
        """ Wrap Dataset properties. """
        if attr in ['n_X', 'n_y', 'X', 'y', 'select_X', 'select_y', 'select_subjects']:
            return getattr(self.dataset, attr)
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, attr))


    def save(self, path='.', prefix='', key=True, summary=True, pickle=False, sep='_'):
        """ Save Measure information to file(s).
        Args:
            path (string): folder to write to.
            prefix (string): all files will be prepended with this.
            sep (string): separator between prefix and rest of filenames.
            key (bool): when True, saves scoring key.
            summary(bool): when True, saves a text summary of Measure.
            picke (bool): when True, pickles the Measure.
        """
        path = os.path.join(path, prefix)
        if prefix != '': path += sep

        if key:
            if not hasattr(self, 'key'):
                raise AttributeError("No scoring key found in current measure. " +
                    "Either add a key, or set key=False in save()")
            np.savetxt(path + 'key.txt', self.key, fmt='%d', delimiter='\t')

        if summary:
            output = str(self)
            open(path + 'summary.txt', 'w').write(output)

        if pickle:
            import pickle
            pickle.dump(self, open(path + 'data.pkl', 'wb'))


class AbbreviatedMeasure(object):
    """ A wrapper for the Measure class that stores both the original, unaltered 
    Measure, and an abbreviated copy. """

    def __init__(self, measure, select, abbreviator=None, evaluator=None, stats=True, trim=True):
        self.original = measure
        self.abbreviator = abbreviator
        self.evaluator = evaluator
        self.abbreviator.abbreviate(measure, select)
        key = self.abbreviator.key
        dataset = copy.deepcopy(measure.dataset)
        dataset.select_X(select)
        self.original_items = select
        self.abbreviated = Measure(dataset, key=key, trim=trim)
        if stats:
            self.compute_stats()

    def __getattr__(self, attr):
        return getattr(self.abbreviated, attr)

    def __str__(self):
        orig = str(self.abbreviated)
        orig += "\n\nOriginal measure items kept: " + ', '.join([str(x) for x in self.original_items])
        return orig
        
