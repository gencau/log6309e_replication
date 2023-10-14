import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

log_name = "../logrep/MCV_hdfsPP-sequential.npz.npz"
dataset = np.load(log_name, allow_pickle=True)
x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

samples = 10000

x_train_slice = x_train[:samples,:]
y_train_slice = y_train[10000]

linkage_data = linkage(x_train_slice, method='ward', metric='euclidean')

plt.figure(figsize=(10,7))
dendrogram(linkage_data, truncate_mode='lastp', p=100)
plt.axhline(y=15, color='red', linestyle='--')

plt.savefig('hierachical_clustering_hdfs.png', bbox_inches='tight')


#from scipy.cluster.hierarchy import fcluster
#from sklearn.metrics import silhouette_score

#best_score = 0
#best_t = 0

## Uncomment to search for best t (only once)
#for cutoff in range(15,300):
    # Assuming 'linkage_data' is the output of your hierarchical clustering
#    clusters = fcluster(linkage_data, t=cutoff, criterion='distance')
#    score = silhouette_score(x_train_slice, clusters, metric='euclidean')
#    if (score > best_score):
#        best_score = score
#        best_t = cutoff


#print('Best Silhouette score: ', best_score)
#print('At: ', best_t)

#clusters = fcluster(linkage_data, t=15, criterion='distance')
#x_train_clusters = np.column_stack((x_train_slice, clusters))
