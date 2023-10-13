import sys
sys.path.append('../')
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from models import DecisionTree


log_name = "../logrep/MCV_hdfsPP-sequential.npz.npz"
dataset = np.load(log_name, allow_pickle=True)
x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

# Get a validation dataset for testing purposes by splitting test in two (don't shuffle!)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

# https://towardsdatascience.com/partial-least-squares-f4e6714452a
pls = PLSRegression(n_components=46, scale=True)
pls.fit(x_train, y_train)

best_r2 = 0
best_ncomp = 0

### Uncomment to test auto-search of best number of components (takes time)
#for n_comp in range(1,46):
#    pls = PLSRegression(n_components=n_comp, scale=True)
#    pls.fit(x_train, y_train)
#    preds = pls.predict(x_val)

#    r2 = r2_score(preds, y_val)
#    if r2 > best_r2:
 #       best_r2 = r2
 #       best_ncomp = n_comp

#print("Best R2, Best number of components:")
#print(best_r2, best_ncomp)

# Validate on test set now
best_model = PLSRegression(n_components=5, scale=True)
best_model.fit(x_train, y_train)
test_preds = best_model.predict(x_test)
print("R2 score on test set:")
print(r2_score(y_test, test_preds))

X_train = best_model.transform(x_train)
X_test = best_model.transform(x_test)

model = DecisionTree.DecisionTree()
model.fit(X_train, y_train)

print('Train validation:')
precision, recall, f1, roc = model.evaluate(X_train, y_train)

print('Test validation:')
precision, recall, f1, roc= model.evaluate(X_test, y_test)

np.savez("../logrep/hdfs-PLS.npz", x_train=X_train,
        y_train=y_train, x_test=X_test, y_test=y_test)

## Perform time series split and cross-validation for statistical ranking
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

#Cross Validation Definition
time_split = TimeSeriesSplit(n_splits=10)

#performance metrics
precision = cross_val_score(model.classifier, X_train, y_train, cv=time_split, scoring = 'precision', n_jobs =1)
print(precision)
