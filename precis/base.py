import numpy as np
import pandas as pd
import logging
from precis import stats
from precis import plot
import copy
import os
from six import string_types

logger = logging.getLogger('precis')


class Dataset(object):

    """ Represents X/y data for Measure training and application, plus some
    additional helper methods. Mostly just wraps pandas.

    Args:
        X (str or DataFrame): The item data. Either the name of a text file
            containing item scores, or a pandas DataFrame.
        y (str or DataFrame): The scale scores (optional). Either the name of a
            text file containing scale scores, or a pandas DataFrame or Series.
        sep (str): field separator in data files.
        missing (str): How to handle subjects with missing item scores. Valid
            options are 'drop', 'impute', or None. See process_missing_data()
            for options.
        select_X (list): An optional list of columns to keep in X (discards the
            rest). Useful when computing measure stats for a previously
            abbreviated measures.
        select_y (list): Optional list of columns to keep in y.
        keep_labels (bool): When False, all items in X will be relabeled
            sequentially starting with 1. When True, the original labels in the
            pandas DataFrame will be kept.
    """

    def __init__(self, X, y=None, sep='\t', missing=None, select_X=None,
                 select_y=None, keep_labels=False):
        # Read in data
        if isinstance(X, string_types):
            X = pd.read_csv(X, sep=sep).convert_objects(convert_numeric=True)
            try:
                X = X.drop('sample', axis=1)
            except:
                pass
        elif not hasattr(X, 'columns'):
            X = pd.DataFrame(X)

        if y is not None and isinstance(y, string_types):
            y = pd.read_csv(y, sep=sep).convert_objects(convert_numeric=True)

        self.X = X
        self.y = y

        self.n_subjects = len(self.X)

        # Store item and scale counts
        self.n_X = self.X.shape[1]

        self._set_X_labels(keep_labels)

        if select_X is not None:
            self.select_X(select_X, keep_labels)

        if y is not None:
            if hasattr(y, 'columns'):
                self.y_labels = self.y.columns
            else:
                self.y_labels = range(y.shape[1])
            if select_y is not None:
                self.select_y(select_y)
            self.n_y = self.y.shape[1]

        if missing is not None:
            self.process_missing_data(missing)

        # Basic validation:
        # if items and scales have different N's, look for an ID column and
        # keep intersection
        if self.y is not None and self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "Number of subjects in item and scale matrices do not match!")

    def process_missing_data(self, missing='drop'):
        """ Process rows in item array that contain missing values.
        Args:
            missing (str): Method for dealing with missing values. Options:
                'drop': Drop any subjects with at least one missing items
                'impute': Impute the mean for that item across all subjects
        """
        if missing == 'drop':
            inds = pd.notnull(self.X).all(1).nonzero()[0]
            if self.y is not None:
                inds = np.intersect1d(
                    inds, pd.notnull(self.y).all(1).nonzero()[0])
            n_missing = len(self.X) - len(inds)

            if n_missing:
                # Slice and reindex X and y
                self.X = self.X.ix[inds]
                if self.y is not None:
                    self.y = self.y.ix[inds]
                logger.info(
                    'Found and deleted %d subjects with missing data.'
                    % n_missing)

        # Imputation. Note that we don't impute the y values, because these
        # should really be inspected and validated by the user before
        # abbreviating.
        elif missing == 'impute':
            self.X = self.X.apply(lambda x: x.fillna(x.mean()), axis=0)
            # self.y = self.y.apply(lambda x: x.fillna(x.mean()), axis=0)

        self.n_subjects = len(self.X)

    def select_subjects(self, inds):
        ''' Trims X and y data to a subset of subjects.
        Args:
            inds (list or array): indices of subjects to keep.
        '''
        self.X = self.X.iloc[inds, :]
        if self.y is not None:
            self.y = self.y.iloc[inds, :]
        self.n_subjects = len(self.X)

    def select_X(self, cols, keep_labels=True):
        ''' Trims X matrix to only the specified items.
        Args:
            cols (list): Columns/items to retain.
            keep_labels (bool): If True, any existing labels (i.e., column
                names) will be kept. If False, all columns will be renumbered
                sequentially.
        '''
        self.X = self.X.ix[:, cols]
        self.n_X = self.X.shape[1]
        self._set_X_labels(keep_labels=keep_labels)

    def _set_X_labels(self, keep_labels=False):
        ''' Number X labels from 0. '''
        if not keep_labels:
            self.X.columns = [str(i + 1) for i in range(self.n_X)]

    @property
    def X_labels(self):
        return self.X.columns

    def select_y(self, cols):
        ''' Trims y to only the specified items.
        Args:
            cols (list or array): indices to keep.
        '''
        if self.y is None:
            raise ValueError(
                "No y array found in measure; nothing to select from!")
        self.y = self.y.ix[:, cols]
        self.n_y = self.y.shape[1]

    def score(self, key, columns=None, rescale=True):
        ''' Compute y scores from X data and scoring key. Note: will overwrite
        any existing y data.
        Args:
            key (string or DataFrame): The scoring key to use. Either a string
                giving the filename of the scoring key, or a pandas DataFrame.
            columns (list): Optional list of column names for the key.
            rescale (bool): If True, adjusts the total y scores to account for
                the presence of reverse-keyed items.
        '''
        if isinstance(key, string_types):
            key = pd.read_csv(key, sep='\t', header=None).values
        y = np.dot(self.X, key)
        if rescale:
            n_reverse = np.sum(key == -1, axis=0)
            max_val = self.X.values.max()
            inc = n_reverse * (max_val + 1)
            y += inc
        if columns is None:
            columns = self.y.columns if self.y is not None else range(
                y.shape[1])
        self.y = pd.DataFrame(y, columns=columns)
        self.n_y = self.y.shape[1]
        self.y_labels = self.y.columns

    def reverse_items(self, items, max_score=None):
        ''' Reverse scores on the items in the list.
        Args:
            items (list): Item numbers to reverse. Items should be indexed
                from 1 and not 0--i.e., pass in the number of the item on the
                scale.
            max_score (int): The value of the highest anchor (e.g., on a
                5-point likert, 5.) If no value is passed, use the single
                highest value across the whole matrix.
        '''
        if max_score is None:
            max_score = np.max(self.X)
        self.X.ix[:, items] = max_score - self.X.ix[:, items] + 1


class Measure(object):

    ''' Represents a measure.

    Args:
        dataset (Dataset): Optional dataset to initialize with.
        X (str or DataFrame): Optional item data to pass to Dataset
            initializer.
        y (str or DataFrame): Optional scale scores to pass to Dataset
            initializer.
        key (str, array, or DataFrame): An optional scoring key
            (items x scales)--either the name of a text file, or a numpy array
            or pandas DataFrame.
        trim (bool): When True, drops all X/y columns not used in scoring key.
        kwargs: Additional keyword arguments to pass on to the Dataset
            initializer.
    '''

    def __init__(self, dataset=None, X=None, y=None, key=None, trim=False,
                 **kwargs):
        if dataset is None:
            if X is None:
                raise ValueError(
                    "Either a Dataset or an X matrix must be provided.")
            dataset = Dataset(X, y, **kwargs)
        self.dataset = dataset

        if key is not None:
            self.set_key(key)

        if trim:
            self.trim()

    def trim(self, key=True, data=True):
        ''' Keep only X and y columns that are non-zero in the key.
        Args:
            key (bool): If True, eliminates all-zero rows/cols from key.
            data (bool): If True, applies trimming to X/y data in Dataset.

        '''
        if not hasattr(self, 'key'):
            raise AttributeError("No key found in current Measure, "
                                 "so trimming is not possible.")

        X_keep = np.any(self.key, axis=1)
        y_keep = np.any(self.key, axis=0)

        if key:
            self.key = self.key[X_keep]
            self.key = self.key[:, y_keep]
        if data:
            self.dataset.select_X(X_keep)
            self.dataset.select_y(y_keep)

    def set_key(self, key):
        """ Set the current scoring key.
        Args:
            key: a numpy array, pandas DataFrame, or the filename of a scoring
                key. Key format is items in rows, scales in columns, with no
                index or header.
        """
        if isinstance(key, string_types):
            key = pd.read_csv(key, sep='\t', header=None)
        if isinstance(key, pd.DataFrame):
            key = key.values
        self.key = key

    def score(self, key=None, columns=None, rescale=True):
        ''' Compute y scores from X data and scoring key. Note: will overwrite
        any existing y data.
        Args:
            key: Optional key to use. If passed, replaces any existing key.
            columns (list): Optional list of column names for the key.
            rescale (bool): If True, adjusts the total y scores to account for
                the presence of reverse-keyed items.
        '''
        if key is not None:
            self.set_key(key)
        if self.key is None:
            raise ValueError(
                "No key found in current measure; can't generate scores!")
        self.dataset.score(self.key, columns, rescale)

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
            self.r_squared = (
                np.corrcoef(dataset.y, self.predicted_y,
                            rowvar=0)[0:self.n_y, self.n_y::] ** 2).diagonal()

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
        output.append('Number of subjects: %d' % self.n_subjects)
        # output.append('Items used from original scale: %s' % ', '.join(str(x+1) for x in self.original_items))

        # Human-readable scoring key
        if self.key is not None:

            output.append('\nScoring key:')

            names = self.dataset.y_labels
            item_labels = self.dataset.X_labels

            for s in range(self.n_y):
                item_list = []
                items_used = np.where(self.key[:, s] != 0)[0]
                for i, v in enumerate(items_used):
                    item = item_labels[v]
                    if self.key[v, s] < 0:
                        item += 'R'
                    item_list.append(item)
                output.append('%s (%d items, R^2=%.2f, alpha=%.2f):\t%s' %
                              (names[s], self.n_items_per_scale[s],
                               self.r_squared[s],
                               self.alpha[s], ', '.join(item_list)))

        return '\n'.join(output)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        """ Wrap Dataset properties. """
        return getattr(self.dataset, attr)

    def save(self, path='.', prefix='', key=True, summary=True, pickle=False,
             sep='_'):
        """ Save Measure information to file(s).
        Args:
            path (str): folder to write to.
            prefix (str): all files will be prepended with this.
            sep (str): separator between prefix and rest of filenames.
            key (bool): when True, saves scoring key.
            summary(bool): when True, saves a text summary of Measure.
            picke (bool): when True, pickles the Measure.
        """
        path = os.path.join(path, prefix)
        if prefix != '':
            path += sep

        if key:
            if not hasattr(self, 'key'):
                raise AttributeError("No scoring key found in current measure."
                                     " Either add a key, or set key=False in "
                                     "save()")
            np.savetxt(path + 'key.txt', self.key, fmt='%d', delimiter='\t')

        if summary:
            output = str(self)
            open(path + 'summary.txt', 'w').write(output)

        if pickle:
            import pickle
            pickle.dump(self, open(path + 'data.pkl', 'wb'))

    def plot_scale_correlation_matrix(self, **kwargs):
        """ Convenience wrapper for scale_correlation_matrix() in plot module.
        """
        return plot.scale_correlation_matrix(self, **kwargs)

    def plot_scale_scatter_plot(self, **kwargs):
        """ Convenience wrapper for scale_scatter_plot() in plot module. """
        return plot.scale_scatter_plot(self, **kwargs)


class AbbreviatedMeasure(object):

    """ A wrapper for the Measure class that stores both the original,
    unaltered Measure, and an abbreviated copy.

    Args:
        measure (Measure): a Measure instance representing the original
            measure.
        select (list): a list of item indices in the original measure to be
            retained in the abbreviation
        key: Optional key to use in the abbreviation. If None, the abbreviator
            argument must be provided.
        abbreviator (Abbreviator): an optional Abbreviator instance to use in
            the abbreviation process. If None, the key argument must be
            provided.
        evaluator (Evaluator): optional Evaluator instance to associate with
            the AbbreviatedMeasure.
        stats (bool): if True, computes stats on the new AbbreviatedMeasures
            post-initialization.
        trim (bool): optional argument passed along to Measure initializer.
        keep_original_labels (bool): when True, the printed scoring key for the
            AbbreviatedMeasure will number items according to the original
            measure rather than the abbreviated version. E.g., if abbreviated
            items 1, 2, and 3 correspond to original items 1, 4, and 8, scoring
            keys will show the latter when printed. When False, indices within
            the abbreviated measure's key will be printed.

    """
    def __init__(self, measure, select, key=None, abbreviator=None,
                 evaluator=None, stats=True, trim=False,
                 keep_original_labels=True):
        self.original = measure
        self.abbreviator = abbreviator
        self.evaluator = evaluator

        if self.abbreviator is not None:
            self.abbreviator.abbreviate(measure, select)
            key = self.abbreviator.key
        elif key is None:
            raise ValueError(
                "Either a key or an abbreviator must be provided.")

        dataset = copy.deepcopy(measure.dataset)
        sel_inds = np.where(select)[0]
        self.original_items = [dataset.X_labels[i] for i in sel_inds]
        dataset.select_X(select, keep_labels=keep_original_labels)
        self.abbreviated = Measure(dataset, key=key, trim=trim)

        if stats:
            self.compute_stats()

    def __getattr__(self, attr):
        """ Wrapper around the stored abbreviated Measure; ensures that by
        default, any attribute request not explicitly defined in
        AbbreviatedMeasure will be passed on to the Measure class. """
        return getattr(self.abbreviated, attr)

    def __str__(self):
        """ Returns the string representation of the abbreviated Measure
        instance, appended with a few details about the abbreviation process.
        """
        orig = str(self.abbreviated)
        orig += "\n\nOriginal measure items kept: " + \
            ', '.join([x for x in self.original_items])
        return orig
