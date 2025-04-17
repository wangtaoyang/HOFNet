import os
import multiprocessing
import json
from pathlib import Path
import numpy as np
from openbabel import pybel
import xgboost as xgb
import argparse
from sklearn.metrics import classification_report
import warnings
import pickle
import pandas as pd
import sys
import os
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

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

def worker(cif_path, hof_name):
    """
    Worker 函数用于计算分子指纹，不做超时控制。
    """
    if not os.path.exists(cif_path):
        print(f"文件不存在: {cif_path}")
        return hof_name, None
    
    try:
        fingerprint = calculate_fingerprint(cif_path, hof_name)
        return hof_name, fingerprint
    except Exception as e:
        print(f"计算分子指纹时出错 ({hof_name}): {e}")
        return hof_name, None

def process_fingerprints_parallel(input_file, output_file, files_path, timeout=5):
    """
    并行计算并保存分子指纹到 JSON 文件中，包含超时处理。
    主进程负责超时控制，子进程仅执行任务。
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    hof_names = list(data.keys())
    
    # 准备并行计算
    fingerprints = {}
    
    # 回调函数，用于收集结果
    def collect_result(result):
        hof_name, fingerprint = result
        if fingerprint is not None:
            fingerprints[hof_name] = fingerprint
    
    # 使用 multiprocessing.Pool 并行处理
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        for hof_name in hof_names:
            cif_path = f'{files_path}/{hof_name}.cif'
            # 使用 apply_async 处理任务，传递回调函数
            result = pool.apply_async(worker, (cif_path, hof_name), callback=collect_result)
            results.append(result)
        
        # 等待所有任务完成，设定超时时间
        for result in results:
            try:
                result.get(timeout=timeout)  # 在主进程中应用超时
            except multiprocessing.TimeoutError:
                print(f"计算超时: {result}")
    
    # 将结果保存到输出文件
    with open(output_file, 'w') as f:
        json.dump(fingerprints, f, indent=4)
    print(f"所有分子指纹已保存到 {output_file}")

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process fingerprints in parallel.")
    
    # 添加命令行参数
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file")
    parser.add_argument('--files_path', type=str, required=True, help="Path to the folder containing files")
    parser.add_argument('--timeout', type=int, default=5, help="Timeout value in seconds (default: 5)")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理函数
    process_fingerprints_parallel(args.input_file, args.output_file, args.files_path, args.timeout)

if __name__ == "__main__":
    main()





