import os
import json
import shutil
from pathlib import Path
from subprocess import Popen, DEVNULL
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def get_vf(cifid: str, ZEO_PATH: str, chan_radius: str, probe_radius: str, num_sample: str, cifs_path: Path):
    try:
        cifpath = cifs_path / f"{cifid}.cif"
        shutil.copy2(cifpath, "./tmp")
        process = Popen(
            [ZEO_PATH, "-ha", "MED", "-vol", chan_radius, probe_radius, num_sample, f"./tmp/{cifid}.cif"],
            stdout=DEVNULL, stderr=DEVNULL
        )
        process.wait()
        vf = None
        vol_file_path = Path(f"./tmp/{cifid}.vol")
        if vol_file_path.exists():
            with vol_file_path.open("r") as f:
                content = f.read()  
                match = re.search(r"AV_Volume_fraction:\s*([\d.]+)", content)
                if match:
                    vf = float(match.group(1))  
        
        return vf
    except Exception as e:
        print(f"Error processing {cifid}: {e}")
        return None

def process_json_parallel(input_file, output_file, ZEO_PATH, chan_radius, probe_radius, num_sample, cifs_path, max_workers=4):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # 创建临时目录
    os.makedirs("./tmp", exist_ok=True)
    
    # 结果字典
    results = {}
    cif_ids = list(data.keys())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_vf, cif_id, ZEO_PATH, chan_radius, probe_radius, num_sample, cifs_path): cif_id for cif_id in cif_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CIFs"):
            cif_id = futures[future]
            try:
                vf = future.result()
                if vf is not None:
                    results[cif_id] = vf
            except Exception as e:
                print(f"Error processing {cif_id}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)
    
    shutil.rmtree("./tmp")

def main():
    parser = argparse.ArgumentParser(description="Process CIF files and calculate volume fractions using ZEO++.")
    
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file")
    parser.add_argument('--ZEO_PATH', type=str, required=True, help="Path to the ZEO++ executable")
    parser.add_argument('--cifs_path', type=str, required=True, help="Path to the directory containing CIF files")
    parser.add_argument('--radius', type=str, default="0.5", help="Radius value for the calculation (default: 0.5)")
    parser.add_argument('--num_sample', type=str, default="50000", help="Number of samples for the calculation (default: 50000)")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of workers for parallel processing (default: 4)")

    args = parser.parse_args()

    process_json_parallel(
        input_file=args.input_file,
        output_file=args.output_file,
        ZEO_PATH=args.ZEO_PATH,
        chan_radius=args.radius,
        probe_radius=args.radius,
        num_sample=args.num_sample,
        cifs_path=Path(args.cifs_path),
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()
