#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('/home/moegwjawe1/atec_project/train/lmf/lib/python3.6/site-packages')  # delete when rank

import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import joblib

input_data_path = '/home/admin/workspace/job/input/test.jsonl'
output_predictions_path = '/home/admin/workspace/job/output/predictions.jsonl'

input_data_path = '/mnt/atec/train.jsonl'  # delete when rank
output_predictions_path = './predictions.jsonl'  # delete when rank


def rank_v0(pd_raw_data):
    pred_list = []

    # 2. 数据预处理
    pd_data = pd_raw_data.copy(deep=True)
    pd_data = pd_data.drop(columns=['memo_polish'], axis=1)
    print('drop memo', pd_data.shape)
    # step3，去除取值只有一个值的列，-15col
    drop_list = ['x2', 'x55', 'x91', 'x96', 'x107', 'x184', 'x198', 'x207', 'x209', 'x261', 'x319', 'x384', 'x436',
                 'x452', 'x456']
    pd_data = pd_data.drop(columns=drop_list)
    print('drop col', pd_data.shape)
    # step4，填充缺失值
    pd_data = pd_data.fillna(pd_data.mean())
    logging.warning("finish step 2")

    # 3. 数据转 numpy
    np_data = pd_data.values
    # 去除索引列
    np_data = np_data[:, 1:]
    # 去除文字列，分成 input 和 output，output 降维到一维
    logging.warning("finish step 3")

    # 4. 逻辑回归
    sc = StandardScaler()
    sc.fit(np_data)
    np_data = sc.transform(np_data)
    logging.warning("finish step 4")

    # 5. 模型预测
    model_path_list = [
        ('./rfc_0.003945.model', 0.003945),  # 0.003945
        ('./lightGBM_0.054045.model', 0.054045),  # 0.054045
    ]

    for model_path in model_path_list:
        model = joblib.load(model_path[0])
        model_score = model_path[1]
        pred = model.predict_proba(np_data)
        pred_list.append(
            (pred, model_score)
        )
    return pred_list


def rank_v1(pd_raw_data):
    pred_list = []

    # 2. 数据预处理
    pd_data_rank = pd_raw_data.copy(deep=True)
    print('raw', pd_data_rank.shape)
    # 2.1 去除 label 列
    #     pd_data_rank = pd_data_rank.drop(columns=['label'], axis=1) # delete when rank
    print('drop label', pd_data_rank.shape)
    # 2.2 去除文字列
    pd_data_rank = pd_data_rank.drop(columns=['memo_polish'], axis=1)
    print('drop memo', pd_data_rank.shape)
    # 2.3 去除离群点
    min_max_map = {}
    file = open('min_max_map_v1.txt', 'r')
    for line in file.readlines():
        line = line.strip()
        line = line.split(' ')
        min_max_map[line[0]] = (float(line[1]), float(line[2]))
    file.close()
    columns = pd_data_rank.columns.values.tolist()[1:]  # 去除 id 的所有列
    for col in columns:
        (d_min, d_max) = min_max_map[col]
        #         print(col, d_min, d_max)
        pd_data_rank[pd_data_rank[col] > d_max] = np.nan
        pd_data_rank[pd_data_rank[col] < d_min] = np.nan
    print('drop outliner', pd_data_rank.shape)
    # 2.4 去除取值只有一个值的列
    drop_list = ['x2', 'x8', 'x13', 'x18', 'x25', 'x32', 'x33', 'x42', 'x48', 'x55', 'x68', 'x70', 'x71', 'x72', 'x77',
                 'x79', 'x91', 'x96', 'x97', 'x100', 'x107', 'x110', 'x111', 'x113', 'x121', 'x130', 'x135', 'x141',
                 'x145', 'x152', 'x155', 'x178', 'x184', 'x198', 'x202', 'x207', 'x209', 'x215', 'x224', 'x225', 'x228',
                 'x232', 'x239', 'x248', 'x249', 'x258', 'x261', 'x264', 'x265', 'x277', 'x279', 'x280', 'x281', 'x289',
                 'x292', 'x300', 'x319', 'x325', 'x341', 'x343', 'x347', 'x351', 'x367', 'x371', 'x373', 'x384', 'x385',
                 'x389', 'x394', 'x400', 'x419', 'x427', 'x431', 'x436', 'x441', 'x442', 'x451', 'x452', 'x456', 'x461',
                 'x462', 'x475']
    pd_data_rank = pd_data_rank.drop(columns=drop_list)
    print('drop cols', pd_data_rank.shape)
    logging.warning("finish step 2")

    # 3. 数据转 numpy
    np_data = pd_data_rank.values
    # 去除索引列
    np_data = np_data[:, 1:]
    print('drop index', np_data.shape)
    # 分成 input 和 output，output 降维到一维
    np_data = np_data.astype(np.float64)

    # 5. 模型预测
    model_path_list = [
        ('./lightGBM_v1.model', 0.445087),
    ]

    for model_path in model_path_list:
        model = joblib.load(model_path[0])
        model_score = model_path[1]
        pred = model.predict_proba(np_data)
        pred_list.append(
            (pred, model_score)
        )
    return pred_list


def rank_v2(pd_raw_data):
    pred_list = []

    # 2. 数据预处理
    pd_data_rank = pd_raw_data.copy(deep=True)
    print('raw', pd_data_rank.shape)
    # 2.1 去除 label 列
    #     pd_data_rank = pd_data_rank.drop(columns=['label'], axis=1) # delete when rank
    print('drop label', pd_data_rank.shape)
    # 2.2 去除文字列
    pd_data_rank = pd_data_rank.drop(columns=['memo_polish'], axis=1)
    print('drop memo', pd_data_rank.shape)
    # 2.3 去除离群点
    min_max_map = {}
    file = open('min_max_map_v2.txt', 'r')
    for line in file.readlines():
        line = line.strip()
        line = line.split(' ')
        min_max_map[line[0]] = (float(line[1]), float(line[2]))
    file.close()
    columns = pd_data_rank.columns.values.tolist()[1:]  # 去除 id 的所有列
    for col in columns:
        (d_min, d_max) = min_max_map[col]
        #         print(col, d_min, d_max)
        pd_data_rank[pd_data_rank[col] > d_max] = np.nan
        pd_data_rank[pd_data_rank[col] < d_min] = np.nan
    print('drop outliner', pd_data_rank.shape)
    # 2.4 去除取值只有一个值的列
    drop_list = ['x2', 'x6', 'x8', 'x12', 'x13', 'x16', 'x18', 'x25', 'x32', 'x33', 'x34', 'x35', 'x38', 'x39', 'x42',
                 'x48', 'x50', 'x55', 'x56', 'x57', 'x68', 'x70', 'x71', 'x72', 'x77', 'x79', 'x85', 'x89', 'x91',
                 'x93', 'x95', 'x96', 'x97', 'x98', 'x100', 'x104', 'x107', 'x110', 'x111', 'x113', 'x116', 'x119',
                 'x121', 'x123', 'x128', 'x130', 'x135', 'x136', 'x138', 'x140', 'x141', 'x145', 'x147', 'x148', 'x151',
                 'x152', 'x155', 'x156', 'x160', 'x166', 'x172', 'x178', 'x179', 'x184', 'x195', 'x198', 'x199', 'x202',
                 'x203', 'x204', 'x206', 'x207', 'x209', 'x213', 'x214', 'x215', 'x221', 'x222', 'x223', 'x224', 'x225',
                 'x226', 'x228', 'x231', 'x232', 'x233', 'x239', 'x240', 'x241', 'x242', 'x243', 'x245', 'x248', 'x249',
                 'x250', 'x258', 'x259', 'x261', 'x264', 'x265', 'x266', 'x273', 'x274', 'x277', 'x279', 'x280', 'x281',
                 'x289', 'x290', 'x292', 'x293', 'x300', 'x302', 'x319', 'x322', 'x325', 'x326', 'x330', 'x332', 'x334',
                 'x341', 'x343', 'x347', 'x351', 'x352', 'x354', 'x358', 'x362', 'x365', 'x367', 'x371', 'x372', 'x373',
                 'x378', 'x380', 'x383', 'x384', 'x385', 'x389', 'x394', 'x397', 'x400', 'x406', 'x407', 'x419', 'x421',
                 'x423', 'x424', 'x427', 'x430', 'x431', 'x432', 'x436', 'x440', 'x441', 'x442', 'x443', 'x444', 'x447',
                 'x451', 'x452', 'x453', 'x455', 'x456', 'x457', 'x461', 'x462', 'x475', 'x477']
    pd_data_rank = pd_data_rank.drop(columns=drop_list)
    print('drop cols', pd_data_rank.shape)
    logging.warning("finish step 2")

    # 3. 数据转 numpy
    np_data = pd_data_rank.values
    # 去除索引列
    np_data = np_data[:, 1:]
    print('drop index', np_data.shape)
    # 分成 input 和 output，output 降维到一维
    np_data = np_data.astype(np.float64)

    # 5. 模型预测
    model_path_list = [
        ('./lightGBM_v2.model', 0.052556),
    ]

    for model_path in model_path_list:
        model = joblib.load(model_path[0])
        model_score = model_path[1]
        pred = model.predict_proba(np_data)
        pred_list.append(
            (pred, model_score)
        )
    return pred_list


def rank_v3(pd_raw_data):
    pred_list = []

    # 2. 数据预处理
    pd_data_rank = pd_raw_data.copy(deep=True)
    print('raw', pd_data_rank.shape)
    # 2.1 去除 label 列
    #     pd_data_rank = pd_data_rank.drop(columns=['label'], axis=1) # delete when rank
    print('drop label', pd_data_rank.shape)
    # 2.2 去除文字列
    pd_data_rank = pd_data_rank.drop(columns=['memo_polish'], axis=1)
    print('drop memo', pd_data_rank.shape)
    # 去除 int 列
    drop_int_list = ['x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x11', 'x18', 'x21', 'x22', 'x26', 'x27', 'x28', 'x29', 'x30',
                     'x32', 'x33', 'x35', 'x37', 'x39', 'x40', 'x41', 'x42', 'x43', 'x47', 'x48', 'x50', 'x51', 'x52',
                     'x56', 'x61', 'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x72', 'x74', 'x75',
                     'x76', 'x78', 'x81', 'x83', 'x86', 'x88', 'x89', 'x92', 'x94', 'x96', 'x102', 'x104', 'x105',
                     'x106', 'x109', 'x110', 'x111', 'x112', 'x114', 'x115', 'x116', 'x118', 'x122', 'x128', 'x129',
                     'x130', 'x131', 'x132', 'x133', 'x134', 'x135', 'x136', 'x139', 'x140', 'x141', 'x145', 'x146',
                     'x147', 'x148', 'x151', 'x152', 'x155', 'x156', 'x159', 'x160', 'x162', 'x163', 'x164', 'x166',
                     'x167', 'x172', 'x174', 'x175', 'x178', 'x179', 'x180', 'x182', 'x183', 'x185', 'x187', 'x188',
                     'x190', 'x192', 'x193', 'x194', 'x195', 'x196', 'x202', 'x203', 'x205', 'x206', 'x211', 'x213',
                     'x215', 'x218', 'x220', 'x221', 'x223', 'x224', 'x226', 'x228', 'x232', 'x234', 'x237', 'x239',
                     'x242', 'x243', 'x247', 'x248', 'x249', 'x253', 'x254', 'x255', 'x258', 'x262', 'x263', 'x264',
                     'x265', 'x268', 'x271', 'x273', 'x274', 'x275', 'x278', 'x280', 'x281', 'x283', 'x284', 'x287',
                     'x290', 'x293', 'x294', 'x295', 'x296', 'x297', 'x299', 'x302', 'x304', 'x305', 'x306', 'x307',
                     'x308', 'x314', 'x315', 'x316', 'x317', 'x318', 'x321', 'x322', 'x323', 'x325', 'x326', 'x328',
                     'x329', 'x331', 'x332', 'x338', 'x339', 'x340', 'x341', 'x343', 'x344', 'x345', 'x346', 'x347',
                     'x348', 'x350', 'x352', 'x353', 'x355', 'x357', 'x358', 'x360', 'x362', 'x363', 'x367', 'x370',
                     'x371', 'x375', 'x376', 'x377', 'x382', 'x383', 'x385', 'x386', 'x387', 'x388', 'x389', 'x391',
                     'x393', 'x395', 'x398', 'x400', 'x401', 'x404', 'x407', 'x408', 'x410', 'x411', 'x413', 'x414',
                     'x415', 'x416', 'x418', 'x419', 'x424', 'x425', 'x428', 'x430', 'x431', 'x433', 'x434', 'x437',
                     'x438', 'x440', 'x442', 'x445', 'x446', 'x449', 'x451', 'x453', 'x454', 'x457', 'x460', 'x465',
                     'x466', 'x469', 'x473', 'x475', 'x478']
    pd_data_rank = pd_data_rank.drop(columns=drop_int_list)
    print('drop int', pd_data_rank.shape)
    # 2.3 去除离群点
    min_max_map = {}
    file = open('min_max_map_v3.txt', 'r')
    for line in file.readlines():
        line = line.strip()
        line = line.split(' ')
        min_max_map[line[0]] = (float(line[1]), float(line[2]))
    file.close()
    columns = pd_data_rank.columns.values.tolist()[1:]  # 去除 id 的所有列
    for col in columns:
        (d_min, d_max) = min_max_map[col]
        #         print(col, d_min, d_max)
        pd_data_rank[pd_data_rank[col] > d_max] = np.nan
        pd_data_rank[pd_data_rank[col] < d_min] = np.nan
    print('drop outliner', pd_data_rank.shape)
    # 2.4 去除取值只有一个值的列
    drop_list = ['x2', 'x12', 'x13', 'x25', 'x55', 'x57', 'x71', 'x77', 'x79', 'x91', 'x97', 'x100', 'x107', 'x113',
                 'x121', 'x138', 'x184', 'x198', 'x204', 'x207', 'x209', 'x225', 'x240', 'x261', 'x266', 'x277', 'x279',
                 'x289', 'x292', 'x300', 'x319', 'x351', 'x372', 'x373', 'x384', 'x394', 'x397', 'x406', 'x427', 'x436',
                 'x441', 'x444', 'x452', 'x455', 'x456', 'x461', 'x462']
    pd_data_rank = pd_data_rank.drop(columns=drop_list)
    print('drop cols', pd_data_rank.shape)
    logging.warning("finish step 2")

    # 3. 数据转 numpy
    np_data = pd_data_rank.values
    # 去除索引列
    np_data = np_data[:, 1:]
    print('drop index', np_data.shape)
    # 分成 input 和 output，output 降维到一维
    np_data = np_data.astype(np.float64)

    # 5. 模型预测
    model_path_list = [
        ('./lightGBM_v0.5_test_[0.859804, 0.900510, 0.200000, LGBMClassifier(verbose=1)].model', 0.203648),
    ]

    for model_path in model_path_list:
        model = joblib.load(model_path[0])
        model_score = model_path[1]
        pred = model.predict_proba(np_data)
        pred_list.append(
            (pred, model_score)
        )
    return pred_list


def rank():
    logging.warning("begin rank")

    # 1. 读取训练数据进行训练
    pd_raw_data = pd.read_json(input_data_path, encoding='utf-8', lines=True)
    logging.warning("finish step 1")
    pd_raw_data = pd_raw_data.drop(columns=['label'], axis=1)  # delete when rank

    pred_list = []

    # pred_list.extend(rank_v0(pd_raw_data))
    pred_list.extend(rank_v1(pd_raw_data))
    # pred_list.extend(rank_v2(pd_raw_data))
    pred_list.extend(rank_v3(pd_raw_data))

    # 6. 保存结果
    with open(output_predictions_path, 'w') as fp:
        for i in range(pd_raw_data.shape[0]):
            label = 0
            score = 0
            for pred in pred_list:
                label += pred[0][i][1] * pred[1] * pred[1]
                # print(pred[0][i][1], pred[1])
                score += pred[1] * pred[1]
            label /= score
            # print("label:", label)
            fp.write('{"id": "%d", "label": %f}\n' % (i, label))
            logging.warning('{"id": "%d", "label": %f}\n' % (i, label))

    logging.warning("end rank")

    return True


if __name__ == '__main__':
    if rank():
        # 训练成功一定要以0方式退出
        sys.exit(0)
    else:
        # 否则-1退出
        sys.exit(1)
