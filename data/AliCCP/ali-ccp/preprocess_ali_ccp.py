import re
import os
import gc
import time
import joblib
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

random.seed(2024)
np.random.seed(2024)
sample_skeleton_train_path = './sample_skeleton_train.csv'
common_features_train_path = './common_features_train.csv'
sample_skeleton_test_path = './sample_skeleton_test.csv'
common_features_test_path = './common_features_test.csv'
save_path = "./"
write_features_map_path = save_path + 'features_map.pkl'
write_features_path = save_path + 'all_features'
sparse_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
uses_columns = [col for col in sparse_columns]


def preprocess_data(mode='train'):
    assert mode in ['train', 'test']
    # if mode == "test" and not os.path.exists(write_features_map_path):
    #     print("Error! Please run the train mode first!")
    #     return
    common_features_path = common_features_train_path if mode == "train" else common_features_test_path
    sample_skeleton_path = sample_skeleton_train_path if mode == "train" else sample_skeleton_test_path

    print(f"Start processing common_features_{mode}")
    common_feat_dict = {}
    with open(common_features_path, 'r') as fr:
        for line in tqdm(fr):
            line_list = line.strip().split(',')
            feat_strs = line_list[2]
            feat_dict = {}
            for fstr in feat_strs.split('\x01'):
                filed, feat_val = fstr.split('\x02')
                feat, val = feat_val.split('\x03')
                if filed in sparse_columns:
                    feat_dict[filed] = feat
            common_feat_dict[line_list[0]] = feat_dict

    print('join feats...')

    with open(f"{write_features_path}_{mode}.tmp", 'w') as fw:
        fw.write('click,purchase,' + ','.join(uses_columns) + '\n')
        with open(sample_skeleton_path, 'r') as fr:
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                if line_list[1] == '0' and line_list[2] == '1':
                    continue
                feat_strs = line_list[5]
                feat_dict = {}
                for fstr in feat_strs.split('\x01'):
                    filed, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    if filed in sparse_columns:
                        feat_dict[filed] = feat
                feat_dict.update(common_feat_dict[line_list[3]])
                feats = line_list[1:3]
                for k in uses_columns:
                    feats.append(feat_dict.get(k, '0'))
                fw.write(','.join(feats) + '\n')


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100 * (start_mem - end_mem) / start_mem, (time.time() - starttime) / 60))
    gc.collect()
    return df


if __name__ == "__main__":
    preprocess_data(mode='train')
    preprocess_data(mode='test')
    # train_data = reduce_mem(pd.read_csv(f"{write_features_path}_train.tmp"))
    train_data = pd.read_csv(f"{write_features_path}_train.tmp")
    test_data = pd.read_csv(f"{write_features_path}_test.tmp")
    # train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=2024)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=2024)

    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    len_train_data = train_data.shape[0]
    len_val_data = val_data.shape[0]
    len_test_data = test_data.shape[0]
    print(f"train_data : {len_train_data}, val_data: {len_val_data}, test_data:{len_test_data}")
    all_data = pd.concat([train_data, val_data, test_data], axis=0)
    del train_data, val_data, test_data
    gc.collect()
    TARGET = ['click', 'purchase']
    col_name = list(all_data.columns)


    # all_data = reduce_mem(all_data)
    train_data = all_data[:len_train_data]
    val_data = all_data[len_train_data:-len_test_data]
    test_data = all_data[-len_test_data:]
    print("start save all ")

    train_data.to_csv(save_path + "ali_ccp_train.csv", index=False)
    val_data.reset_index(drop=True).to_csv(save_path + "ali_ccp_val.csv", index=False)
    test_data.reset_index(drop=True).to_csv(save_path + "ali_ccp_test.csv", index=False)
    print("complete")