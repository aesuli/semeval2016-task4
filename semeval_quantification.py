# Semeval 2016 task 4 - quantification
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

import argparse
import logging

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC

from quantifier import Quantifier
from read_semeval_sentiment import read_semeval_quantification_classification
from rich_analyzer import get_rich_analyzer


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-r', '--train', help='Additional training only data')
    parser.add_argument('-f', '--folds', help='Number of folds (default: leave one out)', type=int)
    parser.add_argument('-s', '--seed', help='Randomization seed (default: 0)', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Verbose output (default: no)', action='store_true')
    parser.add_argument('-l', '--reference_label', help='Name of label to be quantified', type=str, default='positive')
    parser.add_argument('-t', '--test', help='Test file', required=True)
    parser.add_argument('-c', '--c', help='C value for SVM', type=float, default=1.0)
    parser.add_argument('-k', '--k', help='Number of features to keep', type=int, default=1000)
    parser.add_argument('-o', '--output', help='Output filename prefix', required=True)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    data = read_semeval_quantification_classification(args.input, encoding='windows-1252', verbose=args.verbose)

    if args.train:
        train = read_semeval_quantification_classification(args.train, encoding='windows-1252', verbose=args.verbose)
    else:
        train = None

    test = read_semeval_quantification_classification(args.test, encoding='windows-1252', verbose=args.verbose,
                                                      delimiter='\t')

    X = list()
    y = list()
    for key in data:
        X.extend(data[key][0])
        y.extend(data[key][1])
    if train is not None:
        for key in train:
            X.extend(train[key][0])
            y.extend(train[key][1])

    logging.debug('Training set size %i' % len(y))

    logging.debug('Number of test sets %i' % len(test))

    learner = SVC(C=args.c, kernel='linear', probability=True)

    if args.folds:
        logging.debug('Folds %i' % args.folds)
        quantifier = Quantifier(learner, reference_label=args.reference_label,
                                n_folds=args.folds,
                                seed=args.seed)
    else:
        logging.debug('Leave one out %i' % len(y))
        quantifier = Quantifier(learner, reference_label=args.reference_label,
                                n_folds=len(y),
                                seed=args.seed)

    analyzer = get_rich_analyzer(word_ngrams=[2, 3], char_ngrams=[4])

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer=analyzer)),
        ('tfidf', TfidfTransformer()),
        ('chi2', SelectKBest(chi2, k=args.k)),
        ('clf', quantifier),
    ])

    quantifier = pipeline.fit(X, y)

    true_prevalences = dict()
    results = dict()

    for key in test:
        texts, labels = test[key]
        true_prevalences[key] = labels.count(args.reference_label) / len(labels)
        results[key] = quantifier.predict(texts)

    with open('%sc%f-k%i-cc-D.output' % (args.output, args.c, args.k), 'w', encoding='utf8') as ccfile, \
            open('%sc%f-k%i-acc-D.output' % (args.output, args.c, args.k), 'w', encoding='utf8') as accfile, \
            open('%sc%f-k%i-pcc-D.output' % (args.output, args.c, args.k), 'w', encoding='utf8') as pccfile, \
            open('%sc%f-k%i-pacc-D.output' % (args.output, args.c, args.k), 'w', encoding='utf8') as paccfile:
        topics = list(results)
        topics.sort()
        for topic in topics:
            for result in results[topic]:
                if result[0] == 'CC':
                    print('%s\t%0.3f\t%0.3f' % (topic, result[2], 1 - result[2]), file=ccfile)
                elif result[0] == 'ACC':
                    print('%s\t%0.3f\t%0.3f' % (topic, result[2], 1 - result[2]), file=accfile)
                elif result[0] == 'PCC':
                    print('%s\t%0.3f\t%0.3f' % (topic, result[2], 1 - result[2]), file=pccfile)
                elif result[0] == 'PACC':
                    print('%s\t%0.3f\t%0.3f' % (topic, result[2], 1 - result[2]), file=paccfile)


if __name__ == "__main__":
    main()
