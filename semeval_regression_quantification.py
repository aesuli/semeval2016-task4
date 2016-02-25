# Semeval 2016 task 4 - regression quantification
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
from read_semeval_sentiment import read_semeval_quantification_regression
from regression_quantifier import RegressionQuantifier
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

    data = read_semeval_quantification_regression(args.input, encoding='windows-1252')

    texts = list()
    labels = list()
    topics = list()
    for topic in data:
        topic_texts, topic_labels = data[topic]
        texts.extend(topic_texts)
        labels.extend(topic_labels)
        topics.extend([topic for _ in topic_labels])

    analyzer = get_rich_analyzer(word_ngrams=[2, 3], char_ngrams=[4])

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer=analyzer)),
        ('tfidf', TfidfTransformer()),
        ('sel', SelectKBest(chi2, k=args.k)),
        ('clf', BinaryTreeRegressor(base_estimator=LinearSVC(C=args.c), verbose=False)),
    ])

    _, test_topics, test_texts = read_test_data(args.test, encoding='windows-1252')

    quantifier = RegressionQuantifier(pipeline)

    quantifier.fit(texts, labels, topics)

    quantification = quantifier.predict(test_texts, test_topics)

    sorted_topics = list(quantification)
    sorted_topics.sort()
    with open('%sc%f-k%i-plain-E.output' % (args.output, args.c, args.k), 'w', encoding='utf8') as plainfile, \
            open('%sc%f-k%i-corrected_train-E.output' % (args.output, args.c, args.k), 'w',
                 encoding='utf8') as corrected_trainfile, \
            open('%sc%f-k%i-corrected_test-E.output' % (args.output, args.c, args.k), 'w',
                 encoding='utf8') as corrected_testfile:
        for topic in sorted_topics:
            plain, corrected_train, corrected_test = quantification[topic]
            print(topic, *plain, sep='\t', file=plainfile)
            print(topic, *corrected_train, sep='\t', file=corrected_trainfile)
            print(topic, *corrected_test, sep='\t', file=corrected_testfile)


if __name__ == "__main__":
    main()
