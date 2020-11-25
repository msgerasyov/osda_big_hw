#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

class FCAClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.X_pos_ = X[y == 1]
        self.X_neg_ = X[y == 0]
        # Return the classifier
        return self

    def predict(self, X):

        # Input validation
        X = check_array(X)

        y_pred = []

        #FCA algorithm
        for obj in X:
            pos = 0
            neg = 0
            for pos_obj in self.X_pos_:
                if np.sum(obj == pos_obj) > int(len(pos_obj) * self.threshold):
                    pos += 1
            for neg_obj in self.X_neg_:
                if np.sum(obj == neg_obj) > int(len(neg_obj) * self.threshold):
                    neg += 1

            pos = pos / float(len(self.X_pos_))
            neg = neg / float(len(self.X_neg_))
            if (pos > neg):
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_pred = np.array(y_pred)

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def binarize_column(column, bins=5):
    unique = pd.unique(column).shape[0]
    return pd.cut(column, min(unique, bins))

def load_dataset(df):
    X = df.drop(columns=['target'])
    y = np.array(df['target'])
    bin_X = pd.get_dummies(X.transform(binarize_column))
    return bin_X, y

def calculate_metrics(clf, metrics, X, y, cv=10):
    scores = {}
    for metric in metrics:
        scores[metric] = np.mean(cross_val_score(clf, X, y, cv=cv, scoring=metric))
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    conf_mat = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = conf_mat.ravel()
    scores["true positive"] = tp
    scores["false positive"] = fp
    scores["true negative"] = tn
    scores["false negative"] = fn
    return scores
