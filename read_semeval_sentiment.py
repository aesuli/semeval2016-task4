# Semeval 2016 task 4 - data reader
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
import csv
import sys


def filter_polarity_classification(data):
    texts = list()
    labels = list()
    for text, label in zip(*data):
        if label == 'neutral':
            continue
        texts.append(text)
        labels.append(label)
    return texts, labels


def read_semeval_classification(filename, encoding='utf8', verbose=False):
    examples = dict()
    conflicting = set()
    with open(filename, encoding=encoding) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2] in conflicting:
                if verbose:
                    print('Skipping already conflicting text')
                continue
            if row[2] in examples:
                if row[1] != examples[row[2]]:
                    conflicting.add(row[2])
                    if verbose:
                        print('conflicting', row[2], row[1], examples[row[2]])
                    del examples[row[2]]
                    conflicting.add(row[2])
                continue
            examples[row[2]] = row[1]

    labels = list()
    texts = list()
    for key in examples:
        texts.append(key)
        labels.append(examples[key])
    return texts, labels


def read_semeval_quantification_classification(filename, encoding='utf8', verbose=False, delimiter=None):
    examples = dict()
    conflicting = dict()
    with open(filename, encoding=encoding) as f:
        if delimiter:
            reader = csv.reader(f, delimiter=delimiter)
        else:
            reader = csv.reader(f)
        for row in reader:
            topic = row[1]
            if not topic in examples:
                examples[topic] = dict()
                conflicting[topic] = set()
            if row[3] in conflicting[topic]:
                if verbose:
                    print('Skipping already conflicting text')
                continue
            if row[3] in examples[topic]:
                if row[2] != examples[topic][row[3]]:
                    conflicting[topic].add(row[3])
                    if verbose:
                        print('conflicting', topic, row[3], row[2], examples[topic][row[3]])
                    del examples[topic][row[3]]
                    conflicting[topic].add(row[3])
                continue
            examples[topic][row[3]] = row[2]

    data = dict()
    for key in examples:
        labels = list()
        texts = list()
        for text in examples[key]:
            texts.append(text)
            labels.append(examples[key][text])
        data[key] = (texts, labels)

    return data

def read_semeval_quantification_regression(filename, encoding='utf8', verbose=False, delimiter=None):
    examples = dict()
    conflicting = dict()
    with open(filename, encoding=encoding) as f:
        if delimiter:
            reader = csv.reader(f, delimiter=delimiter)
        else:
            reader = csv.reader(f)
        for row in reader:
            topic = row[1]
            if not topic in examples:
                examples[topic] = dict()
                conflicting[topic] = set()
            if row[3] in conflicting[topic]:
                if verbose:
                    print('Skipping already conflicting text')
                continue
            if row[3] in examples[topic]:
                if row[2] != examples[topic][row[3]]:
                    conflicting[topic].add(row[3])
                    if verbose:
                        print('conflicting', topic, row[3], row[2], examples[topic][row[3]])
                    del examples[topic][row[3]]
                    conflicting[topic].add(row[3])
                continue
            examples[topic][row[3]] = int(row[2])

    data = dict()
    for key in examples:
        labels = list()
        texts = list()
        for text in examples[key]:
            texts.append(text)
            labels.append(examples[key][text])
        data[key] = (texts, labels)

    return data

def read_semeval_regression(filename, encoding='utf8', verbose=False):
    examples = dict()
    conflicting = set()
    with open(filename, encoding=encoding) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row[3].strip()) == 0 or len(row[2].strip()) == 0:
                continue
            if row[3] in conflicting:
                if verbose:
                    print('Skipping already conflicting text')
                continue
            if row[3] in examples:
                if row[2] != examples[row[3]]:
                    conflicting.add(row[3])
                    if verbose:
                        print('conflicting', row[3], row[2], examples[row[3]])
                    del examples[row[3]]
                    conflicting.add(row[3])
                continue
            examples[row[3]] = row[2]

    texts = list()
    labels = list()
    for key in examples:
        texts.append(key)
        labels.append(int(examples[key]))

    return texts, labels


def main():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-r', '--regression', help='Read regression data (default: classification)',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Enable verbose output', action='store_true')
    parser.add_argument('-q', '--quantification', help='Read quantification data (default: document-by-document)',
                        action='store_true')
    parser.add_argument('-e', '--encoding', help='Encoding to use (default: utf8, try also windows-1252)',
                        default='utf8')
    args = parser.parse_args()

    if args.regression:
        data = read_semeval_regression(args.input, encoding=args.encoding, verbose=args.verbose)
        print(len(data[0]), len(data[1]))
    elif args.quantification:
        data = read_semeval_quantification_classification(args.input, encoding=args.encoding, verbose=args.verbose)
        for key in data:
            print(key, len(data[key][0]), len(data[key][1]))
    else:
        data = read_semeval_classification(args.input, encoding=args.encoding, verbose=args.verbose)
        print(len(data[0]), len(data[1]))
        data = filter_polarity_classification(data)
        print(len(data[0]), len(data[1]))


if __name__ == "__main__":
    main()
