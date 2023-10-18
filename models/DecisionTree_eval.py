# Acknowledgement:
# Some of the codes are adapted from Loglizer project(https://github.com/logpai/loglizer) 
# and anomaly detection project: (https://github.com/mooselab/suppmaterial-LogRepForAnomalyDetection)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_graphviz
from graphviz import Source
import matplotlib.pyplot as plt
import pydot

def metrics(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_score = roc_auc_score(y_true, y_pred)
    return precision, recall, f1, roc_score



seq_level_data = np.load('../logrep/MCV_hdfsPP-sequential.npz.npz', allow_pickle=True)

x_train = seq_level_data["x_train"]
y_train = seq_level_data["y_train"]
x_test = seq_level_data["x_test"]
y_test = seq_level_data["y_test"]
feature_names = seq_level_data["feature_names"]

##############################
#        Decision Tree
##############################
from sklearn import tree

class DecisionTree(object):

    def __init__(self, criterion='gini', max_depth=None, max_features=None, class_weight=None):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See DecisionTreeClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                          max_features=max_features, class_weight=class_weight)

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
        print('====== Decision Tree: Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1, roc = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, roc-auc: {:.3f}\n'.format(precision, recall, f1, roc))
        return precision, recall, f1, roc

prec_l = []
recall_l = []
f1_l = []
roc_l = []
for i in range(5):
    model = DecisionTree()
    model.classifier.feature_names_in_ = feature_names
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1, roc = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1, roc = model.evaluate(x_test, y_test)
    prec_l.append(precision)
    recall_l.append(recall)
    f1_l.append(f1)
    roc_l.append(roc)

## This is not required for BGL
    if (i == 4): # only print this once
        feat_importance = model.classifier.tree_.compute_feature_importances(normalize=False)
        print("feat importance = " + str(feat_importance))

        for i,v in enumerate(feat_importance):
            print('Feature: %s, Score: %.5f' % (feature_names[i],v))
        tree.export_graphviz(model.classifier, out_file="tree.dot", feature_names=feature_names) 
        
        # Plot them
        plt.barh(range(x_train.shape[1]), feat_importance, tick_label=feature_names)
        plt.ylabel('Features')
        plt.xlabel('Importance')
        plt.tick_params(axis='y', labelsize=5)
        plt.savefig('feature_importance_hdfs.png')


print('average: ', sum(prec_l) / len(prec_l), sum(recall_l) / len(recall_l), sum(f1_l) / len(f1_l), sum(roc_l) / len(roc_l))

## Perform time series split and cross-validation for statistical ranking
## NOT required for BGL
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

#Cross Validation Definition
time_split = TimeSeriesSplit(n_splits=10)

#performance metrics
r2 = cross_val_score(model.classifier, x_train, y_train, cv=time_split, scoring = 'r2', n_jobs =1)
print(r2)