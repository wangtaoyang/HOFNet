import os
import json
import re
import subprocess
import argparse


def process_cif_with_platon(cif_path):
    print(f"Processing {cif_path} with Platon...")
    process = subprocess.Popen(['../PLATON/platon', cif_path], stdin=subprocess.PIPE, text=True)
    process.stdin.write("CALC HBONDS\n")
    process.stdin.write("exit\n")
    process.stdin.close()
    process.wait()
    print(f"Finished processing {cif_path}.")


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

    def create_json_from_cif_list(self, cif_ids, output_json_path):
        hbond_data = {}
        for cif_id in cif_ids:
            cif_path = os.path.join(self.cifs_path, f"{cif_id}.cif")
            if not os.path.exists(cif_path):
                print(f"Skipping {cif_id}: .cif file not found.")
                continue
            process_cif_with_platon(cif_path)
            donors, hs, acceptors = self.get_Hbond_lists(cif_id)
            all_atoms = list(set(donors + hs + acceptors))
            atom_symbols = [atom[0] for atom in all_atoms]
            atom_indices = self.get_atom_indices(cif_id, atom_symbols)
            hbond_data[cif_id] = atom_indices
        with open(output_json_path, 'w') as json_file:
            json.dump(hbond_data, json_file, indent=4)
        print(f"Hydrogen bond JSON written to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate hydrogen bonds from CIFs listed in a JSON file.")
    parser.add_argument('--input_file', type=str, required=True, help="JSON file containing CIF IDs.")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSON file for hydrogen bond indices.")
    parser.add_argument('--cifs_path', type=str, required=True, help="Directory containing .cif files.")

    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        cif_dict = json.load(f)
        cif_ids = list(cif_dict.keys())

    extractor = HbondExtractor(args.cifs_path)
    extractor.create_json_from_cif_list(cif_ids, args.output_file)


if __name__ == "__main__":
    main()
