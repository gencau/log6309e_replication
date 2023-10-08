#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from models import PCA as PCA_
from sklearn.decomposition import PCA
from logrep import preprocessing
from models import DecisionTree
import pandas as pd
import numpy as np

log_name = "../logrep/MCV_hdfsPP-sequential.npz.npz"
dataset = np.load(log_name, allow_pickle=True)
x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

if __name__ == '__main__':
    
    ## Tf-idf term weighting, uncomment this and use hdfsPP-sequential.npz to test
    #feature_extractor = preprocessing.FeatureExtractor()
    #x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
    #                                          normalization='zero-mean')
    #x_test = feature_extractor.transform(x_test)
    
    # Ask for components that make up 99% of the variance of the model
    model = PCA_.PCA(n_components=0.99)
    model.fit(x_train)
    print(model.n_components)

    model = PCA(n_components=model.component_count)
    X_train = model.fit_transform(x_train)
    X_test = model.transform(x_test)
 
    explained_variance = model.explained_variance_ratio_
    print(explained_variance)

    model.fit(X_train)

    print(model.components_)
    print(model.n_components_)

    model = DecisionTree.DecisionTree()
    model.fit(X_train, y_train)

    print('Train validation:')
    precision, recall, f1= model.evaluate(X_train, y_train)

    print('Test validation:')
    precision, recall, f1= model.evaluate(X_test, y_test)

    np.savez("../logrep/hdfs-PCA.npz", x_train=X_train,
         y_train=y_train, x_test=X_test, y_test=y_test)

    ## Perform time series split and cross-validation for statistical ranking
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import cross_val_score

    #Cross Validation Definition
    time_split = TimeSeriesSplit(n_splits=10)

    #performance metrics
    r2 = cross_val_score(model.classifier, X_train, y_train, cv=time_split, scoring = 'r2', n_jobs =1)
    print(r2)
