# ISTI-CNR semeval2016-task4

This is the code used to produce the ISTI-CNR submission to semeval 2016 - task 4.

It provides programs to participate to all the five subtasks:
- A: positive/negative/neutral classification
- B: positive/negative classification
- C: regression on a -2/-1/0/+1/+2 sentiment scale
- D: positive/negative quantification
- E: quantification on a -2/-1/0/+1/+2 sentiment scale

## Regression

With respect to task C, this code implements a scikit-learn compatible learning algorithm that learns a regression model based _data balanced binary tree_ of binary classifiers.
Each node of the tree is a binary classifier that splits the ordinal scale on the point of best balance on training data.
The split process starts on the whole ordinal scale and then it is recursively applied to each part of the split until each single label is reached.

## Quantification

### Binary quantification

With respect to task D, this code implements a scikit-learn like version of the quantification methods presented in:

George Forman,
_"Counting positives accurately despite inaccurate classification."_
Machine Learning: ECML 2005. Springer Berlin Heidelberg, 2005. 564-575.

Antonio Bella et al.,
_"Quantification via probability estimators."_
Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.

- Classify and Count
- Adjusted Classify and Count
- Probabilistic Classify and Count
- Probabilistic Adjusted Classify and Count

### Ordinal scale quantification

With respect to task E, this code implements a novel quantification method _Adjusted Regress and Count_ (ARC), implemented using a scikit-learn like API.
 ARC based on measuring the error of performing regression using a simple Regress and Count method and compensating such error using a linear correction model.

## Citing:

If you use this code to produce results for a paper please insert this citation in your paper (the following paper is currently under review):

Andrea Esuli,
_"ISTI-CNR at SemEval-2016 Task 4: Quantification on an Ordinal Scale"_
Proceedings of the 10th International Workshop on Semantic Evaluation, SemEval '16,
Association for Computational Linguistics

