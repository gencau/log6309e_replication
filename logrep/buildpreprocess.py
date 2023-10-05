import numpy as np
import pandas as pd
import os
from collections import Counter
from dataloader import load_HDFS, load_BGL

# hdfs
data_dir = "../dataset/HDFS_result/"
log_name = "HDFS.log_structured.csv"
label_name = "HDFS.anomaly_label.csv"

# BGL
bgl_data_dir = "../dataset/BGL_result/"
bgl_log_name = "BGL.log_structured.csv"

(x_train, y_train), (x_test, y_test) = load_BGL(bgl_data_dir+bgl_log_name)
np.savez("bglPP.npz",x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)

# 11 min for hdfs
(x_train, y_train), (x_test, y_test) = load_HDFS(
    log_file=data_dir+log_name, label_file=data_dir+label_name)

np.savez("hdfsPP.npz", x_train=x_train,
         y_train=y_train, x_test=x_test, y_test=y_test)
