# Acknowledgement:
# Some of the codes are adapted from Loglizer project(https://github.com/logpai/loglizer) 
# and anomaly detection project: (https://github.com/mooselab/suppmaterial-LogRepForAnomalyDetection)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn import svm

def metrics(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_score = roc_auc_score(y_true, y_pred)
    return precision, recall, f1, roc_score

class SVM(object):

    def __init__(self, penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, 
                 max_iter=100):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See SVM API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        
        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = svm.LinearSVC(penalty=penalty, tol=tol, C=C, dual=dual, 
                                        class_weight=class_weight, max_iter=max_iter)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1, roc = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, roc: {:.3f}\n'.format(precision, recall, f1, roc))
        return precision, recall, f1, roc
