import json
from pathlib import Path
import numpy as np
from openbabel import pybel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
import warnings
import pickle
import pandas as pd
import sys
import os
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer
import concurrent.futures
import csv

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr

def calculate_fingerprint(cif_path, hof_name):
    print(f"开始计算指纹: {hof_name}")
    with suppress_stdout_stderr():
        mol = next(pybel.readfile("cif", cif_path))
        fp = mol.calcfp(fptype='FP2')

    bits_fp = [0] * 1024
    for bit in fp.bits:
        if bit < 1024:
            bits_fp[bit] = 1
    print(f"完成计算指纹: {hof_name}")
    return bits_fp

def get_fingerprint(hof_name, timeout=5):
    cif_path = f'/data/user2/wty/HOF/moftransformer/data/HOF_solvent/cifs/{hof_name}.cif'
    
    if not os.path.exists(cif_path):
        print(f"文件不存在: {cif_path}")
        return None
    
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
    with open(file_path, 'r') as file:
        data = json.load(file)

    with open(fingerprints_file, 'r') as fp_file:
        fingerprints_data = json.load(fp_file)

    features, labels = [], []
    
    for cif_id, label in data.items():
        fingerprint = fingerprints_data.get(cif_id)
        if fingerprint is not None:
            features.append(fingerprint)
            labels.append(label)

    return np.array(features), np.array(labels)

def evaluate_model(train_file, test_file, fingerprints_file, model_save_path, fold_idx):
    X_train, y_train = load_data(train_file, fingerprints_file)
    X_test, y_test = load_data(test_file, fingerprints_file)

    # 使用一个MLP模型预测五元属性
    model = MLPRegressor(hidden_layer_sizes=(512, 256), max_iter=500, random_state=42)
    model.fit(X_train, y_train)  # 训练MLP模型

    predictions = model.predict(X_test)  # 在测试集上进行预测

    # 保存模型
    with open(f"{model_save_path}/model_fold_{fold_idx}.pkl", 'wb') as f:
        pickle.dump(model, f)

    # 保存预测结果到 CSV
    results_df = pd.DataFrame({
        'cif_id': list(json.load(open(test_file)).keys()),
        'true_labels': list(y_test),
        'predictions': list(predictions)
    })
    results_df.to_csv(f"{model_save_path}/results_fold_{fold_idx}.csv", index=False)

    return predictions

def average_results(folder_path):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_data.append(data)
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
    if isinstance(result, dict):
        return {key: convert_ndarray_to_list(value) for key, value in result.items()}
    elif isinstance(result, list):
        return [convert_ndarray_to_list(item) for item in result]
    elif isinstance(result, np.ndarray):
        return result.tolist()
    else:
        return result

def generate_key_value_json(test_file, result_test, output_json):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    key_value_pairs = {}
    for idx, (key, value) in enumerate(test_data.items()):
        key_value_pairs[key] = result_test[idx]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(key_value_pairs, f, indent=4, ensure_ascii=False)

def find_top_solvents(logits, solvent_df, top_n=3):
    solvent_features = solvent_df[['LogP', 'Area', 'donors', 'acceptors', 'point']].values

    neigh = NearestNeighbors(n_neighbors=top_n)
    neigh.fit(solvent_features)

    distances, indices = neigh.kneighbors([logits])
    
    top_solvents = solvent_df.iloc[indices[0]]['solvent1_label'].values.tolist()
    
    return top_solvents

def generate_results_csv(test_file, result_test, solvent_csv, output_csv):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    solvent_df = pd.read_csv(solvent_csv)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(['cif_id', 'solvent_classification_logits', 'solvent_classification_labels',
                         'top1_solvent', 'top2_solvent', 'top3_solvent', 'true_solvent'])

        for idx, (cif_id, label) in enumerate(test_data.items()):
            logits = result_test[idx]
            true_label = label

            logits_str = str(logits.tolist())
            true_label_str = str(true_label)

            top_solvents = find_top_solvents(logits, solvent_df)
            true_solvent = find_top_solvents(true_label, solvent_df, top_n=1)[0]

            writer.writerow([cif_id, logits_str, true_label_str,
                             top_solvents[0], top_solvents[1], top_solvents[2], true_solvent])
    

def evaluate_predictions(true_labels, predictions, average_type='macro'):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average=average_type, zero_division=0)
    recall = recall_score(true_labels, predictions, average=average_type, zero_division=0)
    f1 = f1_score(true_labels, predictions, average=average_type, zero_division=0)
    return accuracy, precision, recall, f1

# Top2和Top3的准确率计算
def multi_top_accuracy(true_labels, predictions):
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] in predictions[i]:
            correct += 1
    return correct / len(true_labels)

def main():
    task = 'solvent'
    base_path = f'/data/user2/wty/HOF/ML_method/MLP/{task}/splits'
    log_path = f'/data/user2/wty/HOF/ML_method/MLP/{task}/logs'
    fp_json_path = f'/data/user2/wty/HOF/ML_method/all_fp.json'
    solvent_csv = '/data/user2/wty/HOF/solvent.csv'

    results = []
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    log_path = f"{log_path}/{timestamp}"
    output_csv = f"{log_path}/test_predictions.csv"

    os.makedirs(log_path, exist_ok=True)

    train_file = f'{base_path}/train_{task}.json'
    val_file = f'{base_path}/val_{task}.json'
    test_file = f'{base_path}/test_{task}.json'

    result_train = evaluate_model(train_file, val_file, fp_json_path, log_path, "train")
    result_test = evaluate_model(train_file, test_file, fp_json_path, log_path, "test")

    generate_results_csv(test_file, result_test, solvent_csv, output_csv)

    # 计算Top1准确度和F1分数
    df = pd.read_csv(output_csv)
    top1_accuracy, top1_precision, top1_recall, top1_f1 = evaluate_predictions(df['true_solvent'], df['top1_solvent'])

    # 由于Top2和Top3为多选一的情况，我们需要先将标签二值化
    lb = LabelBinarizer()
    lb.fit(df['true_solvent'])
    true_labels_binarized = lb.transform(df['true_solvent'])

    top2_preds = df[['top1_solvent', 'top2_solvent']].values.tolist()
    top3_preds = df[['top1_solvent', 'top2_solvent', 'top3_solvent']].values.tolist()

    top2_accuracy = multi_top_accuracy(df['true_solvent'], top2_preds)
    top3_accuracy = multi_top_accuracy(df['true_solvent'], top3_preds)

    print(f"Top1 Accuracy: {top1_accuracy}, Precision: {top1_precision}, Recall: {top1_recall}, F1: {top1_f1}")
    print(f"Top2 Accuracy: {top2_accuracy}")
    print(f"Top3 Accuracy: {top3_accuracy}")

if __name__ == "__main__":
    main()
