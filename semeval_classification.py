# Semeval 2016 task 4 - classification
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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from read_semeval_sentiment import read_semeval_classification, filter_polarity_classification
from rich_analyzer import get_rich_analyzer


def read_test_data(file, binary, encoding, topic=False):
    ids = list()
    text = list()
    topics = list()
    with open(file, 'r', encoding=encoding) as file:
        for line in file:
            fields = line.split('\t')
            ids.append(fields[0])
            if binary:
                text.append(fields[3].strip())
            else:
                text.append(fields[2].strip())
            if topic:
                topics.append(fields[1].strip())
    return ids, text, topics


def main():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-b', '--binary',
                        help='Polarity classification, i.e., posivitive vs negative (default: posivitive/negative/neutral classification)',
                        action='store_true')
    parser.add_argument('-t', '--test', help='Test file', required=True)
    parser.add_argument('-o', '--output', help='Output filename prefix', required=True)
    parser.add_argument('-c', '--c', help='C value for SVM', type=float, default=1.0)
    parser.add_argument('-k', '--k', help='Number of features to keep', type=int, default=1000)
    args = parser.parse_args()

    data = read_semeval_classification(args.input, encoding='windows-1252')
    if args.binary:
        data = filter_polarity_classification(data)

    analyzer = get_rich_analyzer(word_ngrams=[2, 3], char_ngrams=[4])

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer=analyzer)),
        ('tfidf', TfidfTransformer()),
        ('sel', SelectKBest(chi2, k=args.k)),
        ('clf', LinearSVC(C=args.c)),
    ])

    pipeline.fit(data[0], data[1])

    test = read_test_data(args.test, args.binary, encoding='windows-1252', topic=args.binary)

    classifier = pipeline.fit(data[0], data[1])

    y = classifier.predict(test[1])

    if args.binary:
        task = 'B'
    else:
        task = 'A'

    with open('%sc%f-k%i-%s.output' % (args.output, args.c, args.k, task), 'w', encoding='utf8') as outfile:
        if args.binary:
            for id_, topic, label in zip(test[0], test[2], y):
                print(id_, topic, label, sep='\t', file=outfile)
        else:
            for id_, label in zip(test[0], y):
                print(id_, label, sep='\t', file=outfile)


if __name__ == "__main__":
    main()
