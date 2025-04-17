# hofnet version 2.0.0
import os
import random
import json
import pickle
import re
import csv

import numpy as np

import torch
from torch.nn.functional import interpolate


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        cifs_path:str,
        split: str,
        nbr_fea_len: int,
        draw_false_grid=True,
        downstream="",
        tasks=[],
        fold='fold0',
        fp_file_path=None
    ):
        """
        Dataset for pretrained HOF.
        Args:
            data_dir (str): where dataset cif files and energy grid file; exist via model.utils.prepare_data.py
            split(str) : train, test, split
            draw_false_grid (int, optional):  how many generating false_grid_data
            nbr_fea_len (int) : nbr_fea_len for gaussian expansion
        """
        super().__init__()
        self.data_dir = data_dir
        self.cifs_path = cifs_path
        self.draw_false_grid = draw_false_grid
        self.split = split
        self.fp_file_path = fp_file_path
        self.fold = fold
        # self.fp_dict = self._load_fp_dict() if self.fp else None

        assert split in {"train", "test", "val"}
        if downstream:
            path_file = os.path.join(data_dir, self.fold, f"{split}_{downstream}.json")
        else:
            path_file = os.path.join(data_dir, self.fold, f"{split}.json")
        print(f"read {path_file}...")

        if not os.path.isfile(path_file):
            raise FileNotFoundError(
                f"{path_file} doesn't exist. Check 'root_dataset' in config"
            )

        dict_target = json.load(open(path_file, "r"))
        self.cif_ids, self.targets = zip(*dict_target.items())

        self.nbr_fea_len = nbr_fea_len

        self.tasks = {}

        for task in tasks:
            if task in ["mtp", "vfp", "moc", "bbc", "fp", "hbond"]:
                path_file = os.path.join(data_dir, self.fold, f"{split}_{task}.json")
                print(f"read {path_file}...")
                assert os.path.isfile(
                    path_file
                ), f"{path_file} doesn't exist in {data_dir}"

                dict_task = json.load(open(path_file, "r"))
                cif_ids, t = zip(*dict_task.items())
                self.tasks[task] = list(t)
                assert self.cif_ids == cif_ids, print(
                    "order of keys is different in the json file"
                )

    def __len__(self):
        return len(self.cif_ids)

    @staticmethod
    def make_grid_data(grid_data, emin=-5000.0, emax=5000, bins=101):
        """
        make grid_data within range (emin, emax) and
        make bins with logit function
        and digitize (0, bins)
        ****
            caution : 'zero' should be padding !!
            when you change bins, heads.MPP_heads should be changed
        ****
        """
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax

        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1

        return new_grid_data

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_)

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cif_id):
        file_grid = os.path.join(self.cifs_path, f"{cif_id}.grid")
        file_griddata = os.path.join(self.cifs_path, f"{cif_id}.griddata16")

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    def get_grid_data(self, cif_id, draw_false_grid=True):
        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data,
                }
            )
        return ret
    
    # def get_Hbond_lists(self, cif_id):
    #     donors, hs, acceptors = [], [], []
    #     lis_path = os.path.join(self.cifs_path, f"{cif_id}.lis")
    #     # 假如没有lis文件直接返回空list
    #     if not os.path.exists(lis_path):
    #         print(f"No LIS file found for CIF ID {cif_id}.")
    #         return donors, hs, acceptors
    #     with open(lis_path, 'r') as file:
    #         content = file.read()
    #         # find hbond data block
    #         data_block_match = re.search(r"(Nr Typ Res Donor.*?)(?=\n[A-Z])", content, re.DOTALL | re.MULTILINE)
    #     if data_block_match:
    #         data_block = data_block_match.group(0)
    #         lines = data_block.splitlines()
    #         for line in lines:
    #             # 假如line中有？则直接跳过
    #             if "?" in line:
    #                 continue
    #             line = re.sub(r'Intra', ' ', line)
    #             line = re.sub(r'\d\*', '1 ', line)
    #             line = re.sub(r'_[a-z*]', ' ', line)
    #             line = re.sub(r'_[0-9*]', ' ', line)
    #             line = re.sub(r'_', ' ', line)
    #             line = re.sub(r'>', ' ', line)
    #             line = re.sub(r'<', ' ', line)
    #             columns = line.split()
    #             if len(columns) > 1 and (columns[0].isdigit() or columns[0].startswith('**')) and columns[1].isdigit():  # 检查每行是否以数字开头
    #                 # 提取“元素符号+数字”格式
    #                 donor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[2])
    #                 h = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[3])
    #                 acceptor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[4])
    #                 # 将匹配到的结果添加到列表中, 并且donor不以C开头
    #                 if donor and not donor.group().startswith('C'):
    #                     donors.append(donor.group())
    #                     if h:
    #                         hs.append(h.group())
    #                     if acceptor:
    #                         acceptors.append(acceptor.group())
    #         # checker hbond
    #         if len(donors) != len(acceptors):
    #             print('donors:', donors)
    #             print('hs:', hs)
    #             print('acceptors:', acceptors)
    #             print(f"Error in {cif_id}: Donor, H, Acceptor lists have different lengths.")
    #         # else:
    #             # print("No data block found in {}".format(lis_path))
    #     return donors, hs, acceptors   
    
    # @staticmethod
    # def read_cif_extract_block(file_path):
    #     # 从CIF文件中读取内容，并定位到_atom_site_occupancy以下的数据块
    #     with open(file_path, 'r') as file:
    #         content = file.read()
    #     start = content.find('_atom_site_occupancy')
    #     if start == -1:
    #         return None  # 没有找到相应的标签
    #     data_block = content[start:].split('\n')[1:]  # 跳过标签行本身，获取其后的内容
    #     return data_block, len(data_block)

    # @staticmethod
    # def extract_atom_labels(data_block):
    #     atom_labels = []
    #     for line in data_block:
    #         parts = line.strip().split()
    #         if len(parts) < 2:
    #             continue
    #         atom_labels.append(parts[1])  # 假设原子标签总是在第二列
    #     return atom_labels
    
    # @staticmethod
    # def classify_atoms(atom_labels, donors, hs, acceptors):
    #     atom_classification = []
    #     for label in atom_labels:
    #         if label in donors:
    #             atom_classification.append(1)
    #         elif label in hs:
    #             atom_classification.append(2)
    #         elif label in acceptors:
    #             atom_classification.append(3)
    #         else:
    #             atom_classification.append(0)
    #     return atom_classification
            
    # def get_Hbond(self, cif_id):
    #     donors, hs, acceptors = self.get_Hbond_lists(cif_id)
    #     file_path = os.path.join(self.cifs_path, f"{cif_id}.cif")
    #     # 读取和处理CIF文件
    #     data_block, data_len = self.read_cif_extract_block(file_path)
    #     if data_block:
    #         atom_labels = self.extract_atom_labels(data_block)
    #         # print("atom_labels len:", len(atom_labels))
    #         atom_classification = self.classify_atoms(atom_labels, donors, hs, acceptors)
    #         # print(atom_classification)  # 打印结果列表
    #         return torch.LongTensor(np.array(atom_classification,dtype=np.int8))
    #     else:
    #         print("No data block found in the CIF file.")
    #         return None
                

    @staticmethod
    def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
        """
        Expands the distance by Gaussian basis
        (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        _filter = np.linspace(
            dmin, dmax, num_step
        )  # = np.arange(dmin, dmax + step, step) with step = 0.2

        return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var**2).float()

    def get_graph(self, cif_id):
        file_graph = os.path.join(self.cifs_path, f"{cif_id}.graphdata")

        graphdata = pickle.load(open(file_graph, "rb"))
        # graphdata = ["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            self.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
        )

        uni_idx = graphdata[4]
        uni_count = graphdata[5]

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
        }
    
    def _load_fp_dict(self):
        json_path = self.fp_file_path
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                fp_dict = json.load(file)
                fp_dict = {key: torch.FloatTensor(value) for key, value in fp_dict.items()}
                return fp_dict
        except FileNotFoundError:
            print(f"Error: File '{json_path}' not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from file '{json_path}'.")
            raise

    # def get_fp(self, cif_id):
    #     if cif_id not in self.fp_dict:
    #         raise KeyError(f"Key '{cif_id}' not found in the fingerprint JSON data.")
        
    #     return {'fp': self.fp_dict[cif_id]}

    def get_tasks(self, index):
        ret = dict()
        for task, value in self.tasks.items():
            ret.update({task: value[index]})

        return ret


    def __getitem__(self, index):
        # print("in getitem")
        ret = dict()
        cif_id = self.cif_ids[index]
        target = self.targets[index]

        ret.update(
            {
                "cif_id": cif_id,
                "target": target,
            }
        )
        ret.update(self.get_grid_data(cif_id, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cif_id))
        ret.update(self.get_tasks(index))
        return ret

    @staticmethod
    def collate(batch, img_size):
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell), target]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data, target]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_idx in enumerate(batch_nbr_idx):
            n_i = nbr_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_idx += base_idx
            base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []

        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                orig = orig[None, None, :, :, :]
            else:
                orig = interpolate(
                    orig[None, None, :, :, :],
                    size=[img_size, img_size, img_size],
                    mode="trilinear",
                    align_corners=True,
                )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():
            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            new_false_grids = []
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                    orig = orig[None, None, :, :, :]
                else:
                    orig = interpolate(
                        orig[None, None, :, :, :],
                        size=[img_size, img_size, img_size],
                        mode="trilinear",
                        align_corners=True,
                    )
                new_false_grids.append(orig)
            new_false_grids = torch.concat(new_false_grids, axis=0)
            dict_batch["false_grid"] = new_false_grids

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch