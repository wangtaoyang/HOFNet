# hofnet version 2.1.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def compute_regression(pl_module, batch, normalizer):
    infer = pl_module.infer(batch)

    logits = pl_module.regression_head(infer["cls_feats"]).squeeze(-1)  # [B]
    labels = torch.FloatTensor(batch["target"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1

    # normalize encode if config["mean"] and config["std], else pass
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "regression_loss": loss,
        "regression_logits": normalizer.decode(logits),
        "regression_labels": normalizer.decode(labels),
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_regression_loss")(ret["regression_loss"])
    mae = getattr(pl_module, f"{phase}_regression_mae")(
        mean_absolute_error(ret["regression_logits"], ret["regression_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"regression/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"regression/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_classification(pl_module, batch):
    # 前向推理
    infer = pl_module.infer(batch)

    logits, binary = pl_module.classification_head(infer["cls_feats"])  # [B, output_dim]
    labels = torch.LongTensor(batch["target"]).to(logits.device)  # [B]
    
    # 计算 loss
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
    else:
        loss = F.cross_entropy(logits, labels)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classification_loss")(ret["classification_loss"])
    acc = getattr(pl_module, f"{phase}_classification_accuracy")(ret["classification_logits"], ret["classification_labels"])
    # auc_metric = getattr(pl_module, f"{phase}_classification_auc")
    # auc_metric.update(ret["classification_logits"], ret["classification_labels"])
    

    if pl_module.write_log:
        pl_module.log(f"classification/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"classification/{phase}/accuracy", acc, sync_dist=True)
        # pl_module.log(f"classification/{phase}/auc", auc, sync_dist=True)

    return ret

def compute_solvent(pl_module, batch):
    infer = pl_module.infer(batch)
    logits = pl_module.solvent_head(infer["cls_feats"]) # [B, output_dim]
    labels = torch.tensor(batch["target"], dtype=torch.float32).to(logits.device) # [B, output_dim]
    
    loss = F.mse_loss(logits, labels)
    mae =  mean_absolute_error(logits, labels)
    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "atom_num": infer["atom_num"],
        # "solvent_hbond_loss": infer["hbond_loss"],
        "solvent_classification_loss": loss,
        "solvent_classification_logits": logits,
        "solvent_classification_labels": labels,
        "solvent_classification_mae": mae,
        "attn_weights": infer["attn_weights"],
        "rand_idx_list": infer["rand_idx_list"]
    }
    phase = "train" if pl_module.training else "val"
    # attributes = dir(pl_module)
    # print(attributes)
    loss = getattr(pl_module, f"{phase}_solvent_classification_loss")(
        ret["solvent_classification_loss"]
    )
    mae = getattr(pl_module, f"{phase}_solvent_classification_mae")(
        ret["solvent_classification_mae"]
    )
    
    # phase = "train" if pl_module.training else "val"
    # loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    # mae = getattr(pl_module, f"{phase}_vfp_mae")(
    #     mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    # )
    
    if pl_module.write_log:
        pl_module.log(f"solvent_classification/{phase}/mse", loss, sync_dist=True)
        pl_module.log(f"solvent_classification/{phase}/mae", mae, sync_dist=True)

    return ret
    
    

def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=True)

    mpp_logits = pl_module.mpp_head(infer["grid_feats"])  # [B, max_image_len+2, bins]
    mpp_logits = mpp_logits[
        :, :-1, :
    ]  # ignore volume embedding, [B, max_image_len+1, bins]
    mpp_labels = infer["grid_labels"]  # [B, max_image_len+1, C=1]

    mask = mpp_labels != -100.0  # [B, max_image_len, 1]

    # masking
    mpp_logits = mpp_logits[mask.squeeze(-1)]  # [mask, bins]
    mpp_labels = mpp_labels[mask].long()  # [mask]

    mpp_loss = F.cross_entropy(mpp_logits, mpp_labels)

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mpp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mpp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mtp(pl_module, batch):
    infer = pl_module.infer(batch)
    mtp_logits = pl_module.mtp_head(infer["cls_feats"])  # [B, hid_dim]
    mtp_labels = torch.LongTensor(batch["mtp"]).to(mtp_logits.device)  # [B]

    mtp_loss = F.cross_entropy(mtp_logits, mtp_labels)  # [B]

    ret = {
        "mtp_loss": mtp_loss,
        "mtp_logits": mtp_logits,
        "mtp_labels": mtp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mtp_loss")(ret["mtp_loss"])
    acc = getattr(pl_module, f"{phase}_mtp_accuracy")(
        ret["mtp_logits"], ret["mtp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mtp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mtp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_vfp(pl_module, batch):
    infer = pl_module.infer(batch)

    vfp_logits = pl_module.vfp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    vfp_labels = torch.FloatTensor(batch["vfp"]).to(vfp_logits.device)

    assert len(vfp_labels.shape) == 1

    vfp_loss = F.mse_loss(vfp_logits, vfp_labels)
    ret = {
        "vfp_loss": vfp_loss,
        "vfp_logits": vfp_logits,
        "vfp_labels": vfp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    mae = getattr(pl_module, f"{phase}_vfp_mae")(
        mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"vfp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"vfp/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_ggm(pl_module, batch):
    pos_len = len(batch["grid"]) // 2
    neg_len = len(batch["grid"]) - pos_len
    ggm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    ggm_images = []
    for i, (bti, bfi) in enumerate(zip(batch["grid"], batch["false_grid"])):
        if ggm_labels[i] == 1:
            ggm_images.append(bti)
        else:
            ggm_images.append(bfi)

    ggm_images = torch.stack(ggm_images, dim=0)

    batch = {k: v for k, v in batch.items()}
    batch["grid"] = ggm_images

    infer = pl_module.infer(batch)
    ggm_logits = pl_module.ggm_head(infer["cls_feats"])  # cls_feats
    ggm_loss = F.cross_entropy(ggm_logits, ggm_labels.long())

    ret = {
        "ggm_loss": ggm_loss,
        "ggm_logits": ggm_logits,
        "ggm_labels": ggm_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ggm_loss")(ret["ggm_loss"])
    acc = getattr(pl_module, f"{phase}_ggm_accuracy")(
        ret["ggm_logits"], ret["ggm_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"ggm/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"ggm/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_hbond(pl_module, batch):
    if "bbc" in batch.keys():
        task = "bbc"
    else:
        task = "hbond"

    infer = pl_module.infer(batch)
    hbond_logits = pl_module.hbond_head(
        infer["graph_feats"][:, 1:, :]
    ).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    hbond_labels = (
        infer["hbond_labels"].to(hbond_logits).flatten()
    )  # [B, max_graph_len] -> [B * max_graph_len]
    mask = hbond_labels != -100

    hbond_loss = F.binary_cross_entropy_with_logits(
        input=hbond_logits[mask], target=hbond_labels[mask]
    )  # [B * max_graph_len]

    ret = {
        "hbond_loss": hbond_loss,
        "hbond_logits": hbond_logits,
        "hbond_labels": hbond_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret["hbond_loss"])
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        nn.Sigmoid()(ret["hbond_logits"]), ret["hbond_labels"].long()
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_fp(pl_module, batch):
    phase = "train" if pl_module.training else "val"

    infer = pl_module.infer(batch)

    fp_logits = pl_module.fp_head(infer["cls_feats"])  # [B, 1024]
    fp_labels = torch.FloatTensor(batch["fp"]).to(fp_logits.device)  # [B, 1024]

    fp_loss = F.binary_cross_entropy_with_logits(fp_logits, fp_labels)

    fp_predictions = (torch.sigmoid(fp_logits) > 0.5).float()
    # print("fp_predictions:", torch.sum(fp_predictions))
    # print("fp_labels:", torch.sum(fp_labels))
    fp_predictions = fp_predictions.view(-1)
    fp_labels = fp_labels.view(-1)

    # 计算损失和准确率
    loss = getattr(pl_module, f"{phase}_fp_loss")(fp_loss)
    acc = getattr(pl_module, f"{phase}_fp_accuracy")(
        fp_predictions, fp_labels.long()
    )
    # print("fp_loss:", loss)
    # print("fp_accuracy:", acc)
    ret = {
        "fp_loss": fp_loss,
        "fp_logits": fp_logits,
        "fp_labels": fp_labels,
        "fp_accuracy": acc,
    }

    if pl_module.write_log:
        pl_module.log(f"fp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"fp/{phase}/accuracy", acc, sync_dist=True)

    return ret