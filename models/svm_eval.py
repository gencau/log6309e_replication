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



seq_level_data = np.load('../logrep/hdfs-PCA.npz', allow_pickle=True)

x_train = seq_level_data["x_train"]
y_train = seq_level_data["y_train"]
x_test = seq_level_data["x_test"]
y_test = seq_level_data["y_test"]

"""
The implementation of the SVM model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. Failure Prediction 
        in IBM BlueGene/L Event Logs. IEEE International Conference on Data Mining
        (ICDM), 2007.

"""
class SVM(object):

    def __init__(self, penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, 
                 max_iter=1000):
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

model = SVM()
model.fit(x_train, y_train)

print('Train validation:')
precision, recall, f1, roc = model.evaluate(x_train, y_train)

print('Test validation:')
precision, recall, f1, roc = model.evaluate(x_test, y_test)
