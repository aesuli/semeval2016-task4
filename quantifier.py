# scikit-learn style quantifier
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

# this class implements the following quantification methods:
#  - classify and count CC
#  - adjusted classify and count ACC
#  - probabilistic classify and count PCC
#  - probabilistic adjusted classify and count PACC
# see section 2.2 in:
#    Esuli, Andrea, and Fabrizio Sebastiani.
#    "Optimizing text quantifiers for multivariate loss functions."
#    ACM Transactions on Knowledge Discovery from Data (TKDD) 9.4 (2015): 27.
#    http://dl.acm.org/citation.cfm?id=2700406


import argparse
import logging
from multiprocessing.pool import Pool

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm, linear_model
from sklearn import cross_validation
import numpy as np

from read_semeval_sentiment import read_semeval_quantification_classification


class Quantifier(object):
    def __init__(self, clf=None, reference_label=None, n_folds=10, seed=0):
        if clf is None:
            self._clf = svm.SVC(kernel='linear', probability=True)
        else:
            self._clf = clf
        self._n_folds = n_folds
        self._seed = seed
        self._reference_label = reference_label

    def fit(self, X, y):
        labels = list(set(y))
        if len(labels) != 2:
            raise Exception("A binary setup is required")

        min_count = X.shape[0]
        self._min_label = None
        for label in labels:
            count = list(y).count(label)
            if count <= min_count:
                min_count = count
                self._min_label = label

        if self._reference_label is None:
            self._reference_label = self._min_label

        if not self._reference_label in labels:
            raise Exception("Reference label does not appear in training data")

        if min_count >= self._n_folds:
            cv = StratifiedKFold(y, n_folds=min(X.shape[0], self._n_folds), shuffle=True,
                                                  random_state=self._seed)
        else:
            cv = KFold(X.shape[0], n_folds=min(X.shape[0], self._n_folds), shuffle=True,
                                        random_state=self._seed)

        tp = 0
        fp = 0
        ptp = 0
        pfp = 0

        pool = Pool(processes=10)
        requests = list()
        for train_cv, test_cv in cv:
            requests.append((X, y, train_cv, test_cv))

        for fold_tp, fold_fp, fold_ptp, fold_pfp in pool.map(self._fit_fold, requests):
            tp += fold_tp
            fp += fold_fp
            ptp += fold_ptp
            pfp += fold_pfp

        pool.close()

        positives = min_count
        negatives = X.shape[0] - positives
        self._tpr = tp / positives
        self._fpr = fp / negatives
        self._ptpr = ptp / positives
        self._pfpr = pfp / negatives
        self._clf.fit(X, y)
        if self._clf.classes_[0] == self._min_label:
            self._pos_idx = 0
            self._neg_idx = 1
        else:
            self._neg_idx = 0
            self._pos_idx = 1

    def _fit_fold(self, args):
        X, y, train_cv, test_cv = args
        tp = 0
        fp = 0
        ptp = 0
        pfp = 0
        train_X = X[train_cv]
        train_y = np.array([y[i] for i in train_cv])
        test_X = X[test_cv]
        test_y = np.array([y[i] for i in test_cv])
        self._clf.fit(train_X, train_y)
        if self._clf.classes_[0] == self._min_label:
            self._pos_idx = 0
            self._neg_idx = 1
        else:
            self._neg_idx = 0
            self._pos_idx = 1
        predicted_y = self._clf.predict(test_X)
        for (true_label, predicted_label) in zip(test_y, predicted_y):
            if true_label == predicted_label and true_label == self._min_label:
                tp += 1
            elif true_label != predicted_label and true_label != self._min_label:
                fp += 1
        probas = self._clf.predict_proba(test_X)
        for (label, proba) in zip(test_y, probas):
            if label == self._min_label:
                ptp += proba[self._pos_idx]
                pfp += proba[self._neg_idx]
        return tp, fp, ptp, pfp

    def predict(self, X):
        observed = X.shape[0]
        predicted_positive = 0
        positive_probability_sum = 0.0

        predicted_y = self._clf.predict(X)

        predicted_positive += list(predicted_y).count(self._min_label)

        probabilities = self._clf.predict_proba(X)
        for probability in probabilities:
            positive_probability_sum += probability[self._pos_idx]

        CC_prevalence = predicted_positive / observed
        ACC_prevalence = max(0.0, min(1.0, (CC_prevalence - self._fpr) / max(0.0001, (self._tpr - self._fpr))))
        PCC_prevalence = positive_probability_sum / observed
        PACC_prevalence = max(0.0, min(1.0, (PCC_prevalence - self._pfpr) / max(0.0001, (
            self._ptpr - self._pfpr))))

        if self._min_label != self._reference_label:
            CC_prevalence = 1 - CC_prevalence
            ACC_prevalence = 1 - ACC_prevalence
            PCC_prevalence = 1 - PCC_prevalence
            PACC_prevalence = 1 - PACC_prevalence

        results = list()
        results.append(('CC', self._reference_label, CC_prevalence))
        results.append(('ACC', self._reference_label, ACC_prevalence))
        results.append(('PCC', self._reference_label, PCC_prevalence))
        results.append(('PACC', self._reference_label, PACC_prevalence))

        return results


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-t', '--train', help='Additional training only data')
    parser.add_argument('-f', '--folds', help='Number of folds (default: 50)', type=int, default=50)
    parser.add_argument('-s', '--seed', help='Randomization seed (default: 0)', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Verbose output (default: no)', action='store_true')
    parser.add_argument('-r', '--reference_label', help='Name of label to be quantified', type=str, default='positive')
    parser.add_argument('-k', '--k', help='Number of features to keep', type=int, default=1000)
    parser.add_argument('-c', '--c', help='C factor for svm (default: 1.0)', type=float, default=1.0)
    parser.add_argument('-l', '--learner', help='Base learner to use (defalut:svm, options: svm, lr)', type=str,
                        default='svm')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    data = read_semeval_quantification_classification(args.input, encoding='windows-1252', verbose=args.verbose)

    if args.train:
        train = read_semeval_quantification_classification(args.train, encoding='windows-1252', verbose=args.verbose)
    else:
        train = None

    print('"Test set","Category","True prevalence","Method","Predicted prevalence"')
    for test_key in data:

        tests = list()
        X = list()
        y = list()
        for key in data:
            if key == test_key:
                tests.append([key, data[key][0], data[key][1]])
            else:
                X.extend(data[key][0])
                y.extend(data[key][1])
        if train is not None:
            for key in train:
                X.extend(train[key][0])
                y.extend(train[key][1])

        logging.debug('Training set size %i' % len(y))

        logging.debug('Number of test sets %i' % len(tests))

        for test in tests:
            logging.debug('Test %s size %i' % (test[0], len(test[1])))

        if args.learner == 'svm':
            learner = svm.SVC(kernel='linear', probability=True, C=args.c)
        else:
            learner = linear_model.LogisticRegression(C=args.c)

        quantifier = Quantifier(learner, reference_label=args.reference_label,
                                n_folds=args.folds,
                                seed=args.seed)

        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('chi2', SelectKBest(chi2, k=args.k)),
            ('clf', quantifier),
        ])

        quantifier = pipeline.fit(X, y)

        true_prevalences = dict()
        results = dict()

        for testname, texts, labels in tests:
            true_prevalences[testname] = labels.count(args.reference_label) / len(labels)
            results[testname] = quantifier.predict(texts)

        for testname in results:
            for result in results[testname]:
                print('"%s","%s",%0.3f,"%s",%0.3f' % (
                    testname, result[1], true_prevalences[testname], result[0], result[2]))


if __name__ == "__main__":
    main()
