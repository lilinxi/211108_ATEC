#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging

sys.path.append('/home/moegwjawe1/atec_project/train/lmf/lib/python3.6/site-packages')

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import joblib

input_data_path = '/home/admin/workspace/job/input/train.jsonl'
output_model_path = '/home/admin/workspace/job/output/train.model'
result_path = '/home/admin/workspace/job/output/result.json'

input_data_path = '/mnt/atec/train.jsonl'

logging.warning("Begin Train.")
# 1. 读取训练数据进行训练
pd_raw_data = pd.read_json(input_data_path, encoding='utf-8', lines=True)
logging.warning("Finish Read.")

# 2. 数据预处理
pd_data = pd_raw_data.copy(deep=True)
# step1, 去除 -1111，-26 row
pd_data = pd_data.drop(pd_data[(pd_data['x0'] == -1111)].index)
# step2，去除 label = -1，-9265 row
pd_data = pd_data.drop(pd_data[(pd_data['label'] == -1)].index)
# step3，去除取值只有一个值的列，-15col
columns = pd_data.columns.values.tolist()
drop_list = []
for col in columns:
    num = len(pd_data[col].unique())
    if num == 1:
        drop_list.append(col)
pd_data = pd_data.drop(columns=drop_list)
# step4，填充缺失值
pd_data = pd_data.fillna(pd_data.mean())

# 3. 数据转 numpy
np_data = pd_data.values
# print(np_data)
# 去除索引列
np_data = np_data[:, 1:]
# 去除文字列，分成 input 和 output，output 降维到一维
np_data_input = np_data[:, :465]
np_data_input = np_data_input.astype(np.float64)
np_data_output = np_data[:, 466:]
np_data_output = np_data_output.astype(np.int64)
np_data_output = np.squeeze(np_data_output, axis=1)

# 4. 数据归一化
sc = StandardScaler()
sc.fit(np_data_input)
np_data_input_std = sc.transform(np_data_input)
X_train, X_test, Y_train, Y_test = train_test_split(np_data_input, np_data_output, test_size=0.2, random_state=0)

#####################################################
logging.warning("Begin Model.")
result = {}  # 保存一些结果
# lightGBM 模型
from lightgbm import LGBMClassifier

# 5.1 模型定义
fixed_params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,
    'boosting': 'gbdt',
    'num_boost_round': 300,
    'early_stopping_rounds': 30
}
search_params = {
    'learning_rate': 0.4,
    'max_depth': 15,
    'num_leaves': 20,
    'feature_fraction': 0.8,
    'subsample': 0.2
}
model = LGBMClassifier(
    objective='binary',
    metric='auc',
    is_unbalance=True,
    boosting='gbdt',
    num_boost_round=300,
    early_stopping_rounds=30,

    learning_rate=0.4,
    max_depth=15,
    num_leaves=20,
    feature_fraction=0.8,
    subsample=0.2,

    verbose=True
)

# 5.2 模型训练
model.fit(X_train, Y_train)
# 5.3 模型评估
result['acc'] = model.score(X_test, Y_test)
# 5.4 模型保存
joblib.dump(model, './lightGBM.model')
# 5.5 结果保存
with open(result_path, 'w') as fp:
    json.dump(result, fp)
#####################################################
