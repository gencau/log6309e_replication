import numpy as np
import pandas as pd

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

log_name = "../logrep/hdfs-PCA.npz"
dataset = np.load(log_name, allow_pickle=True)
x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

print(x_train.shape)

df =  pd.DataFrame(x_train)
df.columns = ['X1','X2','X3','X4']
df['Label'] = y_train

df_test = pd.DataFrame(x_test)
df_test.columns = ['X1','X2','X3','X4']
df_test['Label'] = y_test

df.merge(df_test)
print(df.head(10))

#find VIF using 'Label' as response variable 
y, X = dmatrices('Label ~ X1+X2+X3+X4', data=df, return_type='dataframe')
X = add_constant(df)

#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

print("VIF results")
#view VIF for each explanatory variable 
print(vif)