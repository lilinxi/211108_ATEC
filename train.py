#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json

sys.path.append('/home/moegwjawe1/atec_project/train/lmf/lib/python3.6/site-packages')

import numpy as np
import pandas as pd

from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib

input_data_path = '/home/admin/workspace/job/input/train.jsonl'
output_model_path = '/home/admin/workspace/job/output/lr.model'
result_path = '/home/admin/workspace/job/output/result.json'

def train():
    # 1. 读取训练数据进行训练
    pd_raw_data = pd.read_json(input_data_path, encoding='utf-8', lines=True)

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

    #####################################################
    # 5. 模型训练
    logreg = linear_model.LogisticRegression(C=1e5, max_iter=100, tol=100)
    logreg.fit(np_data_input_std, np_data_output)
    #####################################################

    # 6. 验证模型
    acc = logreg.score(np_data_input_std, np_data_output)

    # 7. 保存模型
    joblib.dump(logreg, output_model_path)

    # 8. 报错结果
    with open(result_path, 'w') as fp:
        # 任务结果信息，可以在训练执行完后查看
        json.dump({"acc": acc}, fp)
    return True


if __name__ == '__main__':
    if train():
        # 训练成功一定要以0方式退出
        sys.exit(0)
    else:
        # 否则-1退出
        sys.exit(1)
