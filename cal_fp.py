import os
import multiprocessing
import json
from openbabel import pybel
import argparse
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    # Temporarily suppress stdout and stderr
    with open(os.devnull, 'w') as fnull:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr

def calculate_fingerprint(cif_path, hof_name):
    """
    Calculate molecular fingerprint (FP2 type), return bit vector (1024 bits)
    """
    print(f"Start fingerprint: {hof_name}")
    with suppress_stdout_stderr():
        mol = next(pybel.readfile("cif", cif_path))
        fp = mol.calcfp(fptype='FP2')

    bits_fp = [0] * 1024
    for bit in fp.bits:
        if bit < 1024:
            bits_fp[bit] = 1
    print(f"Finished fingerprint: {hof_name}")
    return bits_fp

def worker(cif_path, hof_name):
    """
    Worker function to calculate fingerprint without timeout handling
    """
    if not os.path.exists(cif_path):
        print(f"File not found: {cif_path}")
        return hof_name, None
    
    try:
        fingerprint = calculate_fingerprint(cif_path, hof_name)
        return hof_name, fingerprint
    except Exception as e:
        print(f"Error computing fingerprint ({hof_name}): {e}")
        return hof_name, None

def process_fingerprints_parallel(input_file, output_file, cifs_path, timeout=5):
    """
    Compute fingerprints in parallel and save to JSON, with timeout control
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    hof_names = list(data.keys())
    
    fingerprints = {}

    def collect_result(result):
        hof_name, fingerprint = result
        if fingerprint is not None:
            fingerprints[hof_name] = fingerprint

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        for hof_name in hof_names:
            cif_path = f'{cifs_path}/{hof_name}.cif'
            result = pool.apply_async(worker, (cif_path, hof_name), callback=collect_result)
            results.append(result)
        
        for result in results:
            try:
                result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                print(f"Timeout while computing: {result}")
    
    with open(output_file, 'w') as f:
        json.dump(fingerprints, f, indent=4)
    print(f"Fingerprints saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Parallel fingerprint processing")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSON file with HOF names")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSON file to save fingerprints")
    parser.add_argument('--cifs_path', type=str, required=True, help="Directory containing CIF files")
    parser.add_argument('--timeout', type=int, default=5, help="Timeout in seconds for each fingerprint task")

    args = parser.parse_args()
    process_fingerprints_parallel(args.input_file, args.output_file, args.cifs_path, args.timeout)

if __name__ == "__main__":
    main()
