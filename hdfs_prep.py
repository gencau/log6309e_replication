import pandas as pd
import numpy as np
import re
from collections import OrderedDict
import pickle

struct_log = 'HDFS_result\HDFS.log_structured.csv'
label_data = 'HDFS_result\HDFS.anomaly_label.csv'

df_log = pd.read_csv(struct_log, engine='c', na_filter=False, memory_map=True)
df_label = pd.read_csv(label_data, engine='c', na_filter=False, memory_map=True)

print(type(df_log))
print(df_log.shape)
print(df_log.head(5))

# Extract sequence for each block ID
data_dict = OrderedDict()
for idx, row in df_log.iterrows():
    blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
    blkId_set = set(blkId_list)
    for blk_Id in blkId_set:
        if not blk_Id in data_dict:
            data_dict[blk_Id] = []
        data_dict[blk_Id].append(row['EventId'])
data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

print(type(data_df))
print(data_df.shape)
print(data_df.head(5))

# Merge label
label_data_indexed = df_label.set_index('BlockId')
label_dict = label_data_indexed['Label'].to_dict()
data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

data_df.head(5)

# Function to split the data into train/test
def _split_data(x_data, y_data, train_ratio=0.5):
    pos_idx = y_data > 0
    x_pos = x_data[pos_idx]
    y_pos = y_data[pos_idx]
    x_neg = x_data[~pos_idx]
    y_neg = y_data[~pos_idx]
    train_pos = int(train_ratio * x_pos.shape[0])
    train_neg = int(train_ratio * x_neg.shape[0])
    x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
    y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
    x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
    y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])

    return (x_train, y_train), (x_test, y_test)

# Shuffle the data (random shuffling)
data_df = data_df.sample(frac=1).reset_index(drop=True)
data_df.head(5)

# Split train and test data
train_ratio = 0.7
(x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
    data_df['Label'].values, train_ratio)

# Print train/test data summary
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_total = num_train + num_test
num_train_pos = sum(y_train)
num_test_pos = sum(y_test)
num_pos = num_train_pos + num_test_pos

print('Total: {} instances, {} anomaly, {} normal' \
      .format(num_total, num_pos, num_total - num_pos))
print('Train: {} instances, {} anomaly, {} normal' \
      .format(num_train, num_train_pos, num_train - num_train_pos))
print('Test: {} instances, {} anomaly, {} normal\n' \
      .format(num_test, num_test_pos, num_test - num_test_pos))

print('====== x_train (first five lines) ======')
print(x_train[:5])

print('====== y_train (first five lines) ======')
print(y_train[:5])

# Save to file
dataset_dict = {"X_train": x_train, "X_test": x_test, "y_train": y_train, "y_test": y_test}

with open('hdfs_dict.pickle', 'wb') as file:
    pickle.dump(dataset_dict, file)