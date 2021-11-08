#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# sys.path.append('/home/moegwjawe1/atec_project/train/lmf/lib/python3.6/site-packages')

import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import joblib

input_data_path = '/home/admin/workspace/job/input/test.jsonl'
output_predictions_path = '/home/admin/workspace/job/output/predictions.jsonl'

# input_data_path = '/mnt/atec/train.jsonl'
# output_predictions_path = './predictions.jsonl'

model_path_list = [
    ('./lightGBM_(0.200000, 0.896625, 0.847360).model', 0.847360),
    ('./lightGBM_(0.200000, 0.892860, 0.846328).model', 0.846328),
    ('./lightGBM_(0.200000, 0.967742, 0.843234).model', 0.843234),
    ('./lightGBM_(0.500000, 0.873917, 0.840257).model', 0.840257),
]


def rank():
    logging.warning("begin rank")

    # 1. 读取训练数据进行训练
    pd_raw_data = pd.read_json(input_data_path, encoding='utf-8', lines=True)
    logging.warning("finish step 1")

    # 2. 数据预处理
    pd_data = pd_raw_data.copy(deep=True)
    # step3，去除取值只有一个值的列，-15col
    drop_list = ['x2', 'x55', 'x91', 'x96', 'x107', 'x184', 'x198', 'x207', 'x209', 'x261', 'x319', 'x384', 'x436',
                 'x452', 'x456']
    pd_data = pd_data.drop(columns=drop_list)
    # step4，填充缺失值
    pd_data = pd_data.fillna(pd_data.mean())
    logging.warning("finish step 2")

    # 3. 数据转 numpy
    np_data = pd_data.values
    # 去除索引列
    np_data = np_data[:, 1:]
    # 去除文字列，分成 input 和 output，output 降维到一维
    np_data_input = np_data[:, :465]
    np_data_input = np_data_input.astype(np.float64)
    logging.warning("finish step 3")

    # 4. 逻辑回归
    sc = StandardScaler()
    sc.fit(np_data_input)
    np_data_input_std = sc.transform(np_data_input)
    logging.warning("finish step 4")

    # 5. 模型预测
    pred_list = []

    for model_path in model_path_list:
        model = joblib.load(model_path[0])
        model_score = model_path[1]
        pred = model.predict_proba(np_data_input_std)
        pred_list.append(
            (pred, model_score)
        )
    logging.warning("finish step 5")

    # 6. 保存结果
    with open(output_predictions_path, 'w') as fp:
        for i in range(np_data.shape[0]):
            label = 0
            score = 0
            for pred in pred_list:
                label += pred[0][i][1] * pred[1]
                # print(pred[0][i][1], pred[1])
                score += pred[1]
            label /= score
            # print("label:", label)
            fp.write('{"id": "%d", "label": %f}\n' % (i, label))
            logging.warning('{"id": "%d", "label": %f}\n' % (i, label))

    logging.warning("end rank")

    logging.warning("end rank")
    return True


if __name__ == '__main__':
    if rank():
        # 训练成功一定要以0方式退出
        sys.exit(0)
    else:
        # 否则-1退出
        sys.exit(1)
