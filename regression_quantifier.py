# scikit-learn style regression quantifier
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

from collections import defaultdict

from sklearn import clone
from sklearn.svm import LinearSVC

from binary_tree_regressor import BinaryTreeRegressor


class RegressionQuantifier(object):
    def __init__(self, pipeline=None):
        if pipeline is None:
            self._pipeline = BinaryTreeRegressor(base_estimator=LinearSVC(C=100.0))
        else:
            self._pipeline = pipeline

    def fit(self, X, y, groups):
        self._true_global_prevalences = defaultdict(float)
        self._values = list(set(y))
        self._values.sort()
        for rate in self._values:
            self._true_global_prevalences[rate] = y.count(rate) / len(y)
        self._estimated_global_prevalences = defaultdict(float)
        for group in set(groups):
            group_X = [atext for atext, agroup in zip(X, groups) if agroup == group]
            notgroup_X = [atext for atext, agroup in zip(X, groups) if agroup != group]
            notgroup_y = [alabel for alabel, agroup in zip(y, groups) if agroup != group]
            pipclone = clone(self._pipeline)
            model = pipclone.fit(notgroup_X, notgroup_y)
            predictions = model.predict(group_X)

            for rate in self._values:
                self._estimated_global_prevalences[rate] += predictions.count(rate)

        for rate in self._values:
            self._estimated_global_prevalences[rate] /= len(y)

        self._model = self._pipeline.fit(X, y)

    def predict(self, X, groups):
        predictions = self._model.predict(X)
        quantifications = dict()
        test_global_prevalences = defaultdict(float)
        for rate in self._values:
            test_global_prevalences[rate] = predictions.count(rate) / len(X)
        for group in set(groups):
            group_predictions = [prediction for prediction, agroup in zip(predictions, groups) if agroup == group]
            simple_prevanlences = list()
            corrected_prevalences = list()
            test_corrected_prevalences = list()
            for rate in self._values:
                prevalence = group_predictions.count(rate) / len(group_predictions)
                simple_prevanlences.append(prevalence)
                if self._estimated_global_prevalences[rate] != 0:
                    corrected_prevalences.append(
                        prevalence * self._true_global_prevalences[rate] / self._estimated_global_prevalences[rate])
                else:
                    corrected_prevalences.append(prevalence)
                if test_global_prevalences[rate] != 0:
                    test_corrected_prevalences.append(
                        prevalence * self._true_global_prevalences[rate] / test_global_prevalences[rate])
                else:
                    test_corrected_prevalences.append(prevalence)

            cumulative = sum(corrected_prevalences)
            corrected_prevalences = [corrected_prevalence / cumulative for corrected_prevalence in
                                     corrected_prevalences]
            cumulative = sum(test_corrected_prevalences)
            test_corrected_prevalences = [test_corrected_prevalence / cumulative for test_corrected_prevalence in
                                          test_corrected_prevalences]

            quantifications[group] = (simple_prevanlences, corrected_prevalences, test_corrected_prevalences)
        return quantifications
