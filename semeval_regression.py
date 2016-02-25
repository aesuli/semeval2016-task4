# Semeval 2016 task 4 - regression
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
import codecs

import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from binary_tree_regressor import BinaryTreeRegressor
from read_semeval_sentiment import read_semeval_regression
from rich_analyzer import get_rich_analyzer


def read_test_data(file, encoding):
    ids = list()
    topics = list()
    text = list()
    with open(file, 'r', encoding=encoding) as file:
        for line in file:
            fields = line.split('\t')
            ids.append(fields[0])
            topics.append(fields[1])
            text.append(fields[3].strip())
    return ids, topics, text


def main():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-t', '--test', help='Test file', required=True)
    parser.add_argument('-o', '--output', help='Output filename prefix', required=True)
    parser.add_argument('-c', '--c', help='C value for SVM', type=float, default=1.0)
    parser.add_argument('-k', '--k', help='Number of features to keep', type=int, default=1000)
    args = parser.parse_args()

    data = read_semeval_regression(args.input, encoding='windows-1252')

    analyzer = get_rich_analyzer(word_ngrams=[2, 3], char_ngrams=[4])

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer=analyzer)),
        ('tfidf', TfidfTransformer()),
        ('sel', SelectKBest(chi2, k=args.k)),
        ('clf', BinaryTreeRegressor(base_estimator=LinearSVC(C=args.c), verbose=False)),
    ])

    test = read_test_data(args.test, encoding='windows-1252')

    regressor = pipeline.fit(data[0], data[1])

    y = regressor.predict(test[2])

    with open('%sc%f-k%i-C.output' % (args.output, args.c, args.k), 'w', encoding='utf8') as outfile:
        for id_, topic, rate in zip(test[0], test[1], y):
            print(id_, topic, rate, sep='\t', file=outfile)


if __name__ == "__main__":
    main()
