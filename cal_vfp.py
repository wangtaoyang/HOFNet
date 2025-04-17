import os
import json
import shutil
from pathlib import Path
from subprocess import Popen, DEVNULL
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def get_vf(cifid: str, ZEO_PATH: str, chan_radius: str, probe_radius: str, num_sample: str, cif_dir: Path):
    try:
        cifpath = cif_dir / f"{cifid}.cif"
        shutil.copy2(cifpath, "./tmp")
        process = Popen(
            [ZEO_PATH, "-ha", "MED", "-vol", chan_radius, probe_radius, num_sample, f"./tmp/{cifid}.cif"],
            stdout=DEVNULL, stderr=DEVNULL
        )
        process.wait()
        vf = None
        vol_file_path = Path(f"./tmp/{cifid}.vol")
        
        # 打开 .vol 文件并提取 AV_Volume_fraction
        if vol_file_path.exists():
            with vol_file_path.open("r") as f:
                content = f.read()  # 读取整个文件内容
                # 使用正则表达式查找 AV_Volume_fraction 的值
                match = re.search(r"AV_Volume_fraction:\s*([\d.]+)", content)
                if match:
                    vf = float(match.group(1))  # 提取并转换为浮点数
        
        return vf
    except Exception as e:
        print(f"Error processing {cifid}: {e}")
        return None

def process_json_parallel(json_path, output_json_path, ZEO_PATH, chan_radius, probe_radius, num_sample, cif_dir, max_workers=4):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # 创建临时目录
    os.makedirs("./tmp", exist_ok=True)
    
    # 结果字典
    results = {}
    cif_ids = list(data.keys())

    # 使用线程池进行并行计算
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = {executor.submit(get_vf, cif_id, ZEO_PATH, chan_radius, probe_radius, num_sample, cif_dir): cif_id for cif_id in cif_ids}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CIFs"):
            cif_id = futures[future]
            try:
                vf = future.result()
                if vf is not None:
                    results[cif_id] = vf
            except Exception as e:
                print(f"Error processing {cif_id}: {e}")
    
    # 写入结果到新的 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)
    
    # 删除临时目录
    shutil.rmtree("./tmp")

def main():
    parser = argparse.ArgumentParser(description="Process CIF files and calculate volume fractions using ZEO++.")
    
    # 添加命令行参数
    parser.add_argument('--json_path', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output_json_path', type=str, required=True, help="Path to the output JSON file")
    parser.add_argument('--ZEO_PATH', type=str, required=True, help="Path to the ZEO++ executable")
    parser.add_argument('--cif_dir', type=str, required=True, help="Path to the directory containing CIF files")
    parser.add_argument('--radius', type=str, default="0.5", help="Radius value for the calculation (default: 0.5)")
    parser.add_argument('--num_sample', type=str, default="50000", help="Number of samples for the calculation (default: 50000)")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of workers for parallel processing (default: 4)")

    args = parser.parse_args()

    # 调用并行处理函数
    process_json_parallel(
        json_path=args.json_path,
        output_json_path=args.output_json_path,
        ZEO_PATH=args.ZEO_PATH,
        chan_radius=args.radius,
        probe_radius=args.radius,
        num_sample=args.num_sample,
        cif_dir=Path(args.cif_dir),
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()
