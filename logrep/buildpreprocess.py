import numpy as np
import pandas as pd
import os
from collections import Counter
from dataloader import load_HDFS, load_BGL_dl

# hdfs
data_dir = "..\dataset\HDFS_result\\"
log_name = "HDFS.log_structured.csv"
label_name = "HDFS.anomaly_label.csv"

# BGL
bgl_data_dir = "..\dataset\BGL_result\\"
bgl_log_name = "BGL.log_structured.csv"

params = {
    "log_file": "../dataset/BGL_result/BGL.log_structured.csv",
    "time_range": 21599,  # 6 hours
    "train_ratio": None,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": 0, 
}

(x_train_bgl, x_test_bgl), (y_train_bgl, y_test_bgl) = load_BGL_dl(params,bgl_data_dir)
np.savez("bglPP-sequential.npz",x_train=x_train_bgl,y_train=y_train_bgl,x_test=x_test_bgl,y_test=y_test_bgl)

# 11 min for hdfs

# Generate HDFS with time-based (sequential) splitting
#(x_train, y_train), (x_test, y_test) = load_HDFS(
#    log_file=data_dir+log_name, label_file=data_dir+label_name)

#np.savez("hdfsPP-sequential.npz", x_train=x_train,
#         y_train=y_train, x_test=x_test, y_test=y_test)

# Generate HDFS with random (uniform) splitting
#(x_train, y_train), (x_test, y_test) = load_HDFS(
#    log_file=data_dir+log_name, label_file=data_dir+label_name, split_type='random')

#np.savez("hdfsPP-random.npz", x_train=x_train,
#         y_train=y_train, x_test=x_test, y_test=y_test)
