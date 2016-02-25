# Binary tree regressor
# Copyright (C) 2016  Andrea Esuli <andrea@esuli.it>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np


class BinaryTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, verbose=False):
        self.base_estimator = base_estimator
        self.verbose = verbose
        self._fitted_estimator = None

    def fit(self, X, y):
        self._fitted_estimator = self._fit(X, y)
        return self

    def _fit(self, X, y):
        labels = list(set(y))
        labels.sort()
        if len(labels) == 1:
            if self.verbose:
                print('Leaf', labels)
            return labels

        try:
            counts = [y.count(label) for label in labels]
        except AttributeError:
            unique, allcounts = np.unique(y, return_counts=True)
            counts = [allcounts[np.searchsorted(unique, label)] for label in labels]

        total = len(y)
        div = [abs(0.5 - (sum(counts[:i + 1]) / total)) for i in range(0, len(counts))]
        split_point = div.index(min(div))
        split = labels[split_point]
        left_labels = labels[:split_point + 1]
        right_labels = labels[split_point + 1:]
        if self.verbose:
            print('Training:', labels, counts, div, split, left_labels, right_labels)

        bin_y = [label in left_labels for label in y]
        node_estimator = clone(self.base_estimator)
        node_estimator.fit(X, bin_y)

        left_indexes = [i for i, label in enumerate(y) if label in left_labels]
        left_X = X[left_indexes]
        left_y = [label for label in y if label in left_labels]

        right_indexes = [i for i, label in enumerate(y) if label in right_labels]
        right_X = X[right_indexes]
        right_y = [label for label in y if label in right_labels]

        if self.verbose:
            print('Left/right train size:', len(left_y), len(right_y))

        return node_estimator, self._fit(left_X, left_y), self._fit(right_X, right_y)

    def predict(self, X):
        y_pred = list()
        for x in X:
            y_pred.append(self._predict(x, self._fitted_estimator))
        return y_pred

    def _predict(self, x, estimator):
        if len(estimator) == 1:
            return estimator[0]
        if estimator[0].predict(x):
            return self._predict(x, estimator[1])
        else:
            return self._predict(x, estimator[2])
