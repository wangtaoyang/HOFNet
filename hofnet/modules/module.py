# hofnet version 2.1.0
from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from hofnet.modules import objectives, heads, module_utils
from hofnet.modules.cgcnn import GraphEmbeddings
from hofnet.modules.vision_transformer_3d import VisionTransformer3D

from hofnet.modules.module_utils import Normalizer

import numpy as np
from sklearn.metrics import r2_score


class Module(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.max_grid_len = config["max_grid_len"]
        # self.vis = config["visualize"]

        # graph embedding with_unique_atoms
        self.graph_embeddings = GraphEmbeddings(
            atom_fea_len=config["atom_fea_len"],
            nbr_fea_len=config["nbr_fea_len"],
            max_graph_len=config["max_graph_len"],
            hid_dim=config["hid_dim"],
            vis=config["visualize"],
        )
        self.graph_embeddings.apply(objectives.init_weights)

        # token type embeddings
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.fp_mlp_1 = nn.Sequential(
            nn.Linear(config["hid_dim"] + 1024, 1024),  # 从输入维度加宽到1024
            nn.ReLU(),
            nn.Linear(1024, 768),  # 减少维度到768
            nn.ReLU(),
            nn.Linear(768, 512),  # 进一步减少维度
            nn.ReLU(),
            nn.Linear(512, 768),  # 恢复到最终输出维度768
            nn.ReLU(),
            nn.Linear(768, config["hid_dim"])  # 最终输出层，保持768维度
        )

        # set transformer
        self.transformer = VisionTransformer3D(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            mpp_ratio=config["mpp_ratio"],
        )

        self.layer_norms = nn.ModuleList(nn.LayerNorm(config["hid_dim"]) for _ in range(len(self.transformer.blocks)))
        self.hbond_weight = nn.Parameter(torch.tensor(1.0))

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # volume token
        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(objectives.init_weights)

        # pooler
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # ===================== loss =====================
        if config["loss_names"]["ggm"] > 0:
            self.ggm_head = heads.GGMHead(config["hid_dim"])
            self.ggm_head.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_head = heads.MPPHead(config["hid_dim"])
            self.mpp_head.apply(objectives.init_weights)

        if config["loss_names"]["mtp"] > 0:
            self.mtp_head = heads.MTPHead(config["hid_dim"])
            self.mtp_head.apply(objectives.init_weights)

        if config["loss_names"]["vfp"] > 0:
            self.vfp_head = heads.VFPHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        if config["loss_names"]["hbond"] > 0 or config["loss_names"]["bbc"] > 0:
            self.hbond_head = heads.MOCHead(config["hid_dim"])
            # self.hbond_head.apply(objectives.init_weights)

        if config["loss_names"]["fp"] > 0:
            self.fp_head = heads.FPHead(config["hid_dim"])

        # ===================== Downstream =====================
        hid_dim = config["hid_dim"]

        self.current_tasks = []
        
        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        if self.hparams.config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(hid_dim)
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.hparams.config["loss_names"]["classification"] > 0:
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)
            self.current_tasks = ['classification']
        if self.hparams.config["loss_names"]["solvent_classification"] > 0:
            self.solvent_head = heads.SolventHead(hid_dim)
            self.solvent_head.apply(objectives.init_weights)
            self.current_tasks = ['solvent_classification']

        module_utils.set_metrics(self)
        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        self.test_logits = []
        self.test_labels = []
        self.test_cifid = []
        self.write_log = True

    def infer(
        self,
        batch,
        mask_grid=False,
    ):
        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N']
        # hbond = batch["hbond"]  # [N']

        nbr_idx = batch["nbr_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        uni_idx = batch["uni_idx"]  # list [B]
        uni_count = batch["uni_count"]  # list [B]

        grid = batch["grid"]  # [B, C, H, W, D]
        volume = batch["volume"]  # list [B]
        # print("after init")
        if "hbond" in batch.keys():
            hbond = batch["hbond"]  # [B]
        elif "bbc" in batch.keys():
            hbond = batch["bbc"]  # [B]
        else:
            hbond = None
        
        (
            graph_embeds,  # [B, max_graph_len, hid_dim],
            graph_masks,  # [B, max_graph_len],
            hbond_labels,  # if hbond: [B, max_graph_len], else: None
            rand_idx_list
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            hbond=hbond,
        )
        # add class embeds to graph_embeds
        cls_tokens = torch.zeros(len(crystal_atom_idx)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(crystal_atom_idx), 1).to(graph_masks)  # [B, 1]

        graph_embeds = torch.cat(
            [cls_embeds, graph_embeds], dim=1
        )  # [B, max_graph_len+1, hid_dim]
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)  # [B, max_graph_len+1]

        # get grid embeds
        (
            grid_embeds,  # [B, max_grid_len+1, hid_dim]
            grid_masks,  # [B, max_grid_len+1]
            grid_labels,  # [B, grid+1, C] if mask_image == True
        ) = self.transformer.visual_embed(
            grid,
            max_image_len=self.max_grid_len,
            mask_it=mask_grid,
        )

        # add volume embeds to grid_embeds
        volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
        volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)

        grid_embeds = torch.cat(
            [grid_embeds, volume_embeds], dim=1
        )  # [B, max_grid_len+2, hid_dim]
        grid_masks = torch.cat([grid_masks, volume_mask], dim=1)  # [B, max_grid_len+2]

        # add token_type_embeddings
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_masks, device=self.device).long()
        )
        
        grid_embeds = grid_embeds + self.token_type_embeddings(
            torch.ones_like(grid_masks, device=self.device).long()
        )

        co_embeds = torch.cat(
            [graph_embeds, grid_embeds], dim=1
        )  # [B, final_max_len, hid_dim]
        co_masks = torch.cat(
            [graph_masks, grid_masks], dim=1
        )  # [B, final_max_len, hid_dim]

        x = co_embeds

        attn_weights = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
            # if self.vis and i == 11:
            #     attn_weights.append(_attn)

        x = self.transformer.norm(x)
        graph_feats, grid_feats = (
            x[:, : graph_embeds.shape[1]],
            x[:, graph_embeds.shape[1] :],
        )  # [B, max_graph_len, hid_dim], [B, max_grid_len+2, hid_dim]

        cls_feats = self.pooler(x)  # [B, hid_dim]

        ret = {
            "graph_feats": graph_feats,
            "grid_feats": grid_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "graph_masks": graph_masks,
            "grid_masks": grid_masks,
            "grid_labels": grid_labels, 
            "hbond_labels": hbond_labels,  
            "cif_id": cif_id,
            "atom_num": atom_num,
            "attn_weights": attn_weights,
            "rand_idx_list": rand_idx_list
        }

        return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Graph Grid Matching
        if "ggm" in self.current_tasks:
            ret.update(objectives.compute_ggm(self, batch))

        # MOF Topology Prediction
        if "mtp" in self.current_tasks:
            ret.update(objectives.compute_mtp(self, batch))

        # Void Fraction Prediction
        if "vfp" in self.current_tasks:
            ret.update(objectives.compute_vfp(self, batch))

        # Hydrogen Bond Prediction (or Building Block Classfication)
        if "hbond" in self.current_tasks or "bbc" in self.current_tasks:
            ret.update(objectives.compute_hbond(self, batch))
        
        if "fp" in self.current_tasks:
            ret.update(objectives.compute_fp(self, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = Normalizer(self.mean, self.std)
            ret.update(objectives.compute_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.compute_classification(self, batch))
            
         # solvent
        if "solvent_classification" in self.current_tasks:
            ret.update(objectives.compute_solvent(self, batch))
            
        return ret

    
    def on_train_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def training_step(self, batch, batch_idx):
        output = self(batch)
        # for k, v in output.items():
        #     if "loss" in k:
        #         print(f"train/{k}", v)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        module_utils.epoch_wrapup(self)

    def on_validation_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def on_validation_epoch_end(self) -> None:
        module_utils.epoch_wrapup(self)

    def on_test_start(self,):
        module_utils.set_task(self)
    
    def test_step(self, batch, batch_idx):
        output = self(batch)

        if 'regression_logits' in output.keys():
            self.test_logits += output["regression_logits"].tolist()
            self.test_labels += output["regression_labels"].tolist()
        return output

    def on_test_epoch_end(self):
        module_utils.epoch_wrapup(self)

        # calculate r2 score when regression
        if len(self.test_logits) > 1:
            r2 = r2_score(
                np.array(self.test_labels), np.array(self.test_logits)
            )
            self.log(f"test/r2_score", r2, sync_dist=True)
            self.test_labels.clear()
            self.test_logits.clear()

    def configure_optimizers(self):
        return module_utils.set_schedule(self)
    
    def on_predict_start(self):
        self.write_log = False
        module_utils.set_task(self)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        
        if 'classification_logits' in output:
            if self.hparams.config['n_classes'] == 2:
                output['classification_logits_index'] = torch.round(output['classification_logits']).to(torch.int)
            else:
                softmax = torch.nn.Softmax(dim=1)
                output['classification_logits'] = softmax(output['classification_logits'])
                output['classification_logits_index'] = torch.argmax(output['classification_logits'], dim=1)

        output = {
            k: (v.cpu().tolist() if torch.is_tensor(v) else v)
            for k, v in output.items()
            if ('logits' in k) or ('labels' in k) or 'cif_id' == k
        }

        return output
    
    def on_predict_epoch_end(self, *args):
        if hasattr(self, 'predict_attn_weights') and hasattr(self, 'predict_rand_idx_list'):
            save_data = {
                "cif_id": self.cif_id,
                "atom_num": self.atom_num,
                "attn_weights": self.predict_attn_weights,
                "rand_idx_list": self.predict_rand_idx_list
            }
        self.test_labels.clear()
        self.test_logits.clear()

    def on_predict_end(self, ):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            metric, = args
        else:
            raise ValueError('lr_scheduler_step must have metric and optimizer_idx(optional)')

        if pl.__version__ >= '2.0.0':
            scheduler.step()
        else:
            scheduler.step()
