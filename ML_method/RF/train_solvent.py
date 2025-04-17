import json
from pathlib import Path
import numpy as np
from openbabel import pybel
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import warnings
import pickle
import pandas as pd
import sys
import os
from contextlib import contextmanager
import concurrent.futures
import multiprocessing
import csv

@contextmanager
def suppress_stdout_stderr():
    # 打开空设备
    with open(os.devnull, 'w') as fnull:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr

def calculate_fingerprint(cif_path, hof_name):
    """
    计算分子指纹，返回指纹位向量列表
    """
    print(f"开始计算指纹: {hof_name}")
    with suppress_stdout_stderr():
        mol = next(pybel.readfile("cif", cif_path))
        fp = mol.calcfp(fptype='FP2')  # 使用FP2指纹

    bits_fp = [0] * 1024  # 1024位的位向量
    for bit in fp.bits:
        if bit < 1024:  # 确保 bit 不超过索引范围
            bits_fp[bit] = 1
    print(f"完成计算指纹: {hof_name}")
    return bits_fp

def get_fingerprint(hof_name, timeout=5):
    """
    计算分子指纹，带有超时控制和文件检查
    :param hof_name: 分子的名字
    :param timeout: 超时时间（秒）
    :return: 返回指纹位向量列表，如果失败或超时则返回None
    """
    cif_path = f'/data/user2/wty/HOF/moftransformer/data/HOF_solvent/cifs/{hof_name}.cif'
    
    # 检查文件是否存在
    if not os.path.exists(cif_path):
        print(f"文件不存在: {cif_path}")
        return None
    
    # 使用ProcessPoolExecutor添加超时机制
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(calculate_fingerprint, cif_path, hof_name)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"计算超时: {hof_name}")
            return None
        except Exception as e:
            print(f"计算分子指纹时出错 ({hof_name}): {e}")
            return None

def load_data(file_path, fingerprints_file):
    """
    从 JSON 文件加载数据，并从存储的分子指纹文件中读取对应的指纹
    :param file_path: 输入 JSON 文件的路径，包含 cif_id 和标签
    :param fingerprints_file: 存储分子指纹的 JSON 文件
    :return: features 和 labels 组成的 numpy 数组
    """
    # 读取标签文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 读取指纹文件
    with open(fingerprints_file, 'r') as fp_file:
        fingerprints_data = json.load(fp_file)

    features, labels = [], []
    
    # 遍历标签数据并提取对应的分子指纹
    for cif_id, label in data.items():
        fingerprint = fingerprints_data.get(cif_id)
        if fingerprint is not None:  # 确保指纹存在
            features.append(fingerprint)
            labels.append(label)

    return np.array(features), np.array(labels)

def evaluate_model(train_file, test_file, fingerprints_file, model_save_path, fold_idx):
    X_train, y_train = load_data(train_file, fingerprints_file)
    X_test, y_test = load_data(test_file, fingerprints_file)

    models = []
    predictions = []

    # 为每个属性训练一个随机森林模型
    for i in range(5):
        model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=6)
        model.fit(X_train, y_train[:, i])  # 针对第i个属性进行训练
        models.append(model)

        # 在测试集上进行预测
        pred = model.predict(X_test)
        predictions.append(pred)

        # 保存每个模型
        with open(f"{model_save_path}/model_fold_{fold_idx}_attr_{i}.pkl", 'wb') as f:
            pickle.dump(model, f)

    predictions = np.column_stack(predictions)  # 将5个预测结果组合成一个五元属性

    # 保存预测结果到 CSV
    results_df = pd.DataFrame({
        'cif_id': list(json.load(open(test_file)).keys()),
        'true_labels': list(y_test),
        'predictions': list(predictions)
    })
    results_df.to_csv(f"{model_save_path}/results_fold_{fold_idx}.csv", index=False)

    return predictions

def average_results(folder_path):
    # 存储所有读取的数据
    all_data = []

    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_data.append(data)
    # 初始化平均结果字典
    average_data = {}
    keys = all_data[0].keys()
    for key in keys:
        if isinstance(all_data[0][key], dict):
            average_data[key] = {}
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                if metric in all_data[0][key]:
                    metric_values = [fold[key][metric] for fold in all_data if key in fold]
                    average_data[key][metric] = np.mean(metric_values)
        elif isinstance(all_data[0][key], float):
            accuracy_values = [fold[key] for fold in all_data]
            average_data[key] = np.mean(accuracy_values)
    return average_data

def convert_ndarray_to_list(result):
    """将包含ndarray的字典或其他对象转换为JSON可序列化的形式"""
    if isinstance(result, dict):
        return {key: convert_ndarray_to_list(value) for key, value in result.items()}
    elif isinstance(result, list):
        return [convert_ndarray_to_list(item) for item in result]
    elif isinstance(result, np.ndarray):
        return result.tolist()
    else:
        return result

def generate_key_value_json(test_file, result_test, output_json):
    """根据test_file中的key，将result_test的预测结果形成k-v键值对并保存为JSON"""
    # 读取test_solvent.json
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 创建k-v对，key为test_solvent.json中的key，value为模型预测的结果
    key_value_pairs = {}
    for idx, (key, value) in enumerate(test_data.items()):
        # 将模型的预测结果添加为value
        key_value_pairs[key] = result_test[idx]

    # 保存k-v对为JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(key_value_pairs, f, indent=4, ensure_ascii=False)


def find_top_solvents(logits, solvent_df, top_n=3):
    """
    根据给定的模型预测logits，使用最近邻算法寻找最接近的溶剂。
    :param logits: 模型预测的五元溶剂属性
    :param solvent_df: 包含溶剂信息的DataFrame
    :param top_n: 寻找最接近的溶剂数量
    :return: 返回前 top_n 个最接近的溶剂的标签列表
    """
    # 使用溶剂的五元特征：LogP, Area, donors, acceptors, point 进行匹配
    solvent_features = solvent_df[['LogP', 'Area', 'donors', 'acceptors', 'point']].values

    # 使用最近邻算法
    neigh = NearestNeighbors(n_neighbors=top_n)
    neigh.fit(solvent_features)

    # 寻找最近的top_n个溶剂
    distances, indices = neigh.kneighbors([logits])
    
    # 获取最近邻的溶剂标签
    top_solvents = solvent_df.iloc[indices[0]]['solvent1_label'].values.tolist()
    
    return top_solvents

def generate_results_csv(test_file, result_test, solvent_csv, output_csv):
    """
    根据测试集生成包含溶剂预测结果的CSV文件
    :param test_file: 测试集文件路径
    :param result_test: 模型预测结果
    :param solvent_csv: 溶剂信息文件路径
    :param output_csv: 生成的输出CSV文件路径
    """
    # 读取test_solvent.json
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 读取溶剂信息的CSV文件
    solvent_df = pd.read_csv(solvent_csv)

    # 打开CSV文件准备写入
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(['cif_id', 'solvent_classification_logits', 'solvent_classification_labels',
                         'top1_solvent', 'top2_solvent', 'top3_solvent', 'true_solvent'])

        # 遍历每个测试集中的样本
        for idx, (cif_id, label) in enumerate(test_data.items()):
            logits = result_test[idx]  # 模型预测的五种溶剂属性
            true_label = label  # 测试集真实标签

            # 转换logits和true_label为字符串
            logits_str = str(logits.tolist())
            true_label_str = str(true_label)

            # 根据logits找到最接近的三个溶剂
            top_solvents = find_top_solvents(logits, solvent_df)

            # 根据true_label找到最接近的溶剂
            true_solvent = find_top_solvents(true_label, solvent_df, top_n=1)[0]

            # 将数据写入CSV
            writer.writerow([cif_id, logits_str, true_label_str,
                             top_solvents[0], top_solvents[1], top_solvents[2], true_solvent])

def main():
    task = 'solvent'
    base_path = f'/data/user2/wty/HOF/ML_method/RF/{task}/splits'
    log_path = f'/data/user2/wty/HOF/ML_method/RF/{task}/logs'
    fp_json_path = f'/data/user2/wty/HOF/ML_method/all_fp.json'
    solvent_csv = '/data/user2/wty/HOF/solvent.csv'  # 溶剂信息的CSV文件路径

    results = []
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    log_path = f"{log_path}/{timestamp}"
    output_csv = f"{log_path}/test_predictions.csv"

    os.makedirs(log_path, exist_ok=True)

    train_file = f'{base_path}/train_{task}.json'
    val_file = f'{base_path}/val_{task}.json'
    test_file = f'{base_path}/test_{task}.json'

    # 进行训练、验证、测试
    result_train = evaluate_model(train_file, val_file, fp_json_path, log_path, "train")
    result_test = evaluate_model(train_file, test_file, fp_json_path, log_path, "test")

    # # 转换结果中的ndarray为list，以便可以被json序列化
    # result_train_serializable = convert_ndarray_to_list(result_train)
    # result_test_serializable = convert_ndarray_to_list(result_test)

    # 生成结果的CSV文件
    generate_results_csv(test_file, result_test, solvent_csv, output_csv)

if __name__ == "__main__":
    main()
