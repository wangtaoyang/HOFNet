import hofnet
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import hofnet
from hofnet.examples import example_path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json

import numpy as np
import ast  # 用于将字符串解析为列表
from sklearn.metrics import mean_squared_error
import csv

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer
import os
# from hofnet.examples import example_path

# data root and downstream from example
fold = 'fold5'
devices = [1]
max_epochs = 200
batch_size = 32
visualize = False
seed = 0               # default seeds
BASE_DATA = './data'
BASE_LOG = './logs'
root_dataset = f'{BASE_DATA}/HOF_solvent/fold{fold}'
task = 'solvent'
downstream = task
log_dir = f'{BASE_LOG}/solvent/fold{fold}'
os.makedirs(log_dir, exist_ok=True)
cifs_path = './data/total'
load_path = './logs/HOF_pretrain/fold5/pretrained_mof_seed0_from_/version_2/checkpoints/best.ckpt' # 改进后的预训练策略，使用真实hof预训练的模型


if load_path is not None:
    pretrain_model = load_path.split('/')[-1].split('.')[0]
else:
    pretrain_model = ''

def get_latest_version(log_dir, seed):
    base_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_{pretrain_model}'
    os.makedirs(base_path, exist_ok=True)
    version_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('version_')]
    if not version_dirs:
        return -1
    
    latest_version = max(int(d.name.split('_')[1]) for d in version_dirs)
    return latest_version

version = get_latest_version(log_dir, seed) + 1 
print("version:", version)

hofnet.run(root_dataset, downstream, log_dir=log_dir,                   
                max_epochs=max_epochs, batch_size=batch_size, devices=devices, cifs_path=cifs_path,
                loss_names="solvent_classification", num_workers=4, load_path=load_path,
                freeze_layers=False)

data_set = 'HOF_solvent'
losses_name = 'solvent_classification'

root_dataset = f'{BASE_DATA}/{data_set}/fold{fold}'

# Get ckpt file
# For version > 2.1.1, best.ckpt exists
checkpoint = 'best'    # Epochs where the model is stored. 
load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_{pretrain_model}/version_{version}/checkpoints/{checkpoint}.ckpt'
save_dir = Path(log_dir) / f'pretrained_mof_seed{seed}_from_{pretrain_model}/version_{version}'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')

print(f'load_path: {load_path}')
hofnet.predict(root_dataset, load_path, cifs_path=cifs_path, downstream=downstream,
                   save_dir=save_dir, devices=devices, loss_names=losses_name, visualize=visualize)


base_path = f'{BASE_LOG}/{task}/fold{fold}/pretrained_mof_seed0_from_{pretrain_model}'
solvents_df = pd.read_csv('/mnt/user2/wty/HOF/solvent.csv')
# 假设solvents_df具有列'solvent_label'和'properties'

# 读取分类结果数据
results_df = pd.read_csv(Path(base_path) / f'version_{version}/test_prediction.csv')

# 定义一个计算MSE的函数
def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

# 处理预测和实际标签数据，将字符串转换为列表
results_df['solvent_classification_logits'] = results_df['solvent_classification_logits'].apply(ast.literal_eval)
results_df['solvent_classification_labels'] = results_df['solvent_classification_labels'].apply(ast.literal_eval)

# 创建用于存储最接近的三个溶剂标签的空列
results_df['top1_solvent'] = ""
results_df['top2_solvent'] = ""
results_df['top3_solvent'] = ""


# 合并solvents_df中的性质列为一个列表
solvents_df['properties'] = solvents_df[['LogP', 'Area', 'donors', 'acceptors', 'point']].values.tolist()
# 为每个预测计算最接近的三个溶剂标签
for index, row in results_df.iterrows():
    # 计算所有溶剂性质与预测性质之间的MSE
    mse_scores = solvents_df['properties'].apply(lambda x: mse(x, row['solvent_classification_logits']))
    # 找到MSE最小的三个溶剂标签
    closest_solvents = mse_scores.nsmallest(3).index
    results_df.at[index, 'top1_solvent'] = solvents_df.at[closest_solvents[0], 'solvent1_label']
    results_df.at[index, 'top2_solvent'] = solvents_df.at[closest_solvents[1], 'solvent1_label']
    results_df.at[index, 'top3_solvent'] = solvents_df.at[closest_solvents[2], 'solvent1_label']
    
    # find nearest label
    mse_scores_true = solvents_df['properties'].apply(lambda x: mse(x, row['solvent_classification_labels']))
    closest_true_solvent = mse_scores_true.idxmin()
    results_df.at[index, 'true_solvent'] = solvents_df.at[closest_true_solvent, 'solvent1_label']

# 保存结果
results_df.to_csv(Path(base_path) / f'version_{version}/updated_classification_results.csv', index=False)


# 读取CSV文件
df_hof = pd.read_csv(Path(base_path) / f'version_{version}/updated_classification_results.csv')
df_solvents = pd.read_csv('/mnt/user2/wty/HOF/solvent.csv')

# 溶剂性质的列名
properties = ['LogP', 'Area', 'donors', 'acceptors', 'point']

# 读取CSV文件
df = pd.read_csv(f'{BASE_LOG}/{task}/fold{fold}/pretrained_mof_seed0_from_{pretrain_model}/version_{version}/updated_classification_results.csv')

# 准备评估函数
def evaluate_predictions(true_labels, predictions, average_type='macro'):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average=average_type, zero_division=0)
    recall = recall_score(true_labels, predictions, average=average_type, zero_division=0)
    f1 = f1_score(true_labels, predictions, average=average_type, zero_division=0)
    return accuracy, precision, recall, f1

# 计算Top1准确度和F1分数
top1_accuracy, top1_precision, top1_recall, top1_f1 = evaluate_predictions(df['true_solvent'], df['top1_solvent'])

# 由于Top2和Top3为多选一的情况，我们需要先将标签二值化
lb = LabelBinarizer()
lb.fit(df['true_solvent'])
true_labels_binarized = lb.transform(df['true_solvent'])

# Top2和Top3的准确率计算
def multi_top_accuracy(true_labels, predictions):
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] in predictions[i]:
            correct += 1
    return correct / len(true_labels)

top2_preds = df[['top1_solvent', 'top2_solvent']].values.tolist()
top3_preds = df[['top1_solvent', 'top2_solvent', 'top3_solvent']].values.tolist()

top2_accuracy = multi_top_accuracy(df['true_solvent'], top2_preds)
top3_accuracy = multi_top_accuracy(df['true_solvent'], top3_preds)

print(f"Top1 Accuracy: {top1_accuracy}, Precision: {top1_precision}, Recall: {top1_recall}, F1: {top1_f1}")
print(f"Top2 Accuracy: {top2_accuracy}")
print(f"Top3 Accuracy: {top3_accuracy}")