#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from models import PCA as PCA_
from sklearn.decomposition import PCA
from logrep import dataloader, preprocessing
from models import DecisionTree
import pandas as pd
import numpy as np

log_name = "../logrep/hdfsPP-sequential.npz"
dataset = np.load(log_name, allow_pickle=True)
x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

if __name__ == '__main__':
    
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    model = PCA_.PCA()
    model.fit(x_train)
    print(model.n_components)

    # I chose 5 after running the PCA demo with PCA from the models directory (above this block) 
    model = PCA(n_components=5)
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
