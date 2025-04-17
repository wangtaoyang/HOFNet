import subprocess
import os
import json
import re
import argparse
from pathlib import Path

def process_cif_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".cif"):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filepath} with Platon...")
            # 调用 Platon 命令
            process = subprocess.Popen(['../PLATON/platon', filepath], stdin=subprocess.PIPE, text=True)
            # 向 Platon 发送命令
            process.stdin.write("CALC HBONDS\n")
            process.stdin.write("exit\n")
            process.stdin.close()
            # 等待 Platon 命令执行完成
            process.wait()
            print(f"Finished processing {filepath}.")

def remove_empty_lists_from_json(src_json_path, dest_json_path):
    with open(src_json_path, 'r') as src_file:
        data = json.load(src_file)
    
    cleaned_data = {k: v for k, v in data.items() if v != []}
    
    with open(dest_json_path, 'w') as dest_file:
        json.dump(cleaned_data, dest_file, indent=4)
    print(f"Removed empty lists and saved to {dest_json_path}")

class HbondExtractor:
    def __init__(self, cifs_path):
        self.cifs_path = cifs_path

    def get_Hbond_lists(self, cif_id):
        donors, hs, acceptors = [], [], []
        lis_path = os.path.join(self.cifs_path, f"{cif_id}.lis")
        if not os.path.exists(lis_path):
            print(f"No LIS file found for CIF ID {cif_id}.")
            return donors, hs, acceptors
        with open(lis_path, 'r') as file:
            content = file.read()
            data_block_match = re.search(r"(Nr Typ Res Donor.*?)(?=\n[A-Z])", content, re.DOTALL | re.MULTILINE)
        if data_block_match:
            data_block = data_block_match.group(0)
            lines = data_block.splitlines()
            for idx, line in enumerate(lines):
                if "?" in line:
                    continue
                line = re.sub(r'Intra', ' ', line)
                line = re.sub(r'\d\*', '1 ', line)
                line = re.sub(r'_[a-z*]', ' ', line)
                line = re.sub(r'_[0-9*]', ' ', line)
                line = re.sub(r'_', ' ', line)
                line = re.sub(r'>', ' ', line)
                line = re.sub(r'<', ' ', line)
                columns = line.split()
                if len(columns) > 1 and (columns[0].isdigit() or columns[0].startswith('**')) and columns[1].isdigit():
                    donor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[2])
                    h = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[3])
                    acceptor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[4])
                    if donor and not donor.group().startswith('C'):
                        donors.append((donor.group(), idx))
                        if h:
                            hs.append((h.group(), idx))
                        if acceptor:
                            acceptors.append((acceptor.group(), idx))
            if len(donors) != len(acceptors):
                print('donors:', donors)
                print('hs:', hs)
                print('acceptors:', acceptors)
                print(f"Error in {cif_id}: Donor, H, Acceptor lists have different lengths.")
        return donors, hs, acceptors

    def get_atom_indices(self, cif_id, atoms):
        cif_path = os.path.join(self.cifs_path, f"{cif_id}.cif")
        if not os.path.exists(cif_path):
            print(f"No CIF file found for CIF ID {cif_id}.")
            return []
        atom_indices = []
        with open(cif_path, 'r') as file:
            lines = file.readlines()
            atom_block = False
            atom_list_start_index = None
            for idx, line in enumerate(lines):
                if line.strip() == "_atom_site_occupancy":
                    atom_block = True
                    atom_list_start_index = idx + 1
                elif atom_block and line.strip() == "loop_":
                    break
                elif atom_block:
                    columns = line.split()
                    if len(columns) > 1 and columns[0] in atoms:
                        atom_indices.append(idx - atom_list_start_index)
        return atom_indices
    
    def create_json_from_cifs(self, output_json_path):
        hbond_data = {}
        for filename in os.listdir(self.cifs_path):
            if filename.endswith(".cif"):
                cif_id = os.path.splitext(filename)[0]
                donors, hs, acceptors = self.get_Hbond_lists(cif_id)
                all_atoms = list(set(donors + hs + acceptors))
                atom_symbols = [atom[0] for atom in all_atoms]
                atom_indices = self.get_atom_indices(cif_id, atom_symbols)
                hbond_data[cif_id] = atom_indices
        with open(output_json_path, 'w') as json_file:
            json.dump(hbond_data, json_file, indent=4)
        print(f"JSON file created at {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Process CIF files, calculate hydrogen bonds using Platon, and generate JSON.")
    
    parser.add_argument('--cifs_path', type=str, required=True, help="Path to the folder containing CIF files")
    parser.add_argument('--output_json_path', type=str, required=True, help="Path to the output JSON file")

    args = parser.parse_args()

    # Step 1: Process CIF files using Platon to calculate hydrogen bonds
    process_cif_files(args.cifs_path)

    # Step 2: Extract hydrogen bond data and create a JSON file
    extractor = HbondExtractor(args.cifs_path)
    extractor.create_json_from_cifs(args.output_json_path)

if __name__ == "__main__":
    main()
