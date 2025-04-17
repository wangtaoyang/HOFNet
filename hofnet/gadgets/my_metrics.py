import torch
from torchmetrics import Metric
from sklearn.metrics import roc_auc_score


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if len(logits.shape) > 1:
            preds = logits.argmax(dim=-1)
        else:
            # binary accuracy
            logits[logits >= 0.5] = 1
            logits[logits < 0.5] = 0
            preds = logits

        preds = preds[target != -100]
        target = target[target != -100]

        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class MultiClassAUC(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        # 累积所有预测值和目标标签
        self.add_state("preds", default=torch.zeros(0, num_classes), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.zeros(0, num_classes), dist_reduce_fx="cat")

    def update(self, logits, target):
        # 将预测值和目标标签送到当前设备
        logits, target = logits.detach().to(self.preds.device), target.detach().to(self.preds.device)

        # 获取每个类别的概率（softmax 输出）
        probs = torch.softmax(logits, dim=-1)

        # 将目标标签转换为 one-hot 编码
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, target.view(-1, 1), 1)

        # 累积预测值和目标标签
        self.preds = torch.cat((self.preds, probs), dim=0)
        self.targets = torch.cat((self.targets, target_one_hot), dim=0)

    def compute(self):
        # 计算 AUC（在所有批次完成之后）
        if self.preds.numel() == 0 or self.targets.numel() == 0:
            return torch.tensor(0.0, device=self.preds.device)

        # 计算 AUC，使用 sklearn 的 roc_auc_score
        # print(f"y_true shape: {self.targets.squeeze()}")
        # print(f"y_pred shape: {self.preds.squeeze()}")
        # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        num_unique_classes = torch.unique(torch.argmax(self.targets.squeeze(), dim=1)).numel()
        if num_unique_classes == 1:
            return torch.tensor(0.0, device=self.preds.device)

        auc = roc_auc_score(self.targets.squeeze().cpu(), self.preds.squeeze().cpu(), average='micro', multi_class='ovr')
        return torch.tensor(auc, device=self.preds.device)

    def reset(self):
        # 清除累积状态
        self.preds = torch.zeros(0, self.num_classes, device=self.preds.device)
        self.targets = torch.zeros(0, self.num_classes, device=self.targets.device)


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total

class WeightedAccuracy(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes  # 类别数量是外部传入的
        # 初始化累积状态
        self.add_state("correct", default=torch.zeros(num_classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("class_counts", default=torch.zeros(num_classes, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        # 计算类别预测结果
        if len(logits.shape) > 1:
            preds = logits.argmax(dim=-1)
        else:
            # binary accuracy
            logits[logits >= 0.5] = 1
            logits[logits < 0.5] = 0
            preds = logits

        # 过滤掉无效标签
        preds = preds[target != -100]
        target = target[target != -100]

        if target.numel() == 0:
            return

        assert preds.shape == target.shape

        # 更新类别计数
        unique_classes = torch.unique(target)

        # 累加每个类别的样本数（仅限有效类别）
        for c in unique_classes:
            if c < self.num_classes:  # 确保类别不超过预设的类别数
                self.class_counts[c] += torch.sum(target == c)
                self.correct[c] += torch.sum((preds == c) & (target == c))
                self.total[c] += torch.sum(target == c)

    def compute(self):
        if self.class_counts.sum() == 0:
            return torch.tensor(0.0, device=self.correct.device)

        # 计算加权准确率
        weighted_accuracy = 0.0
        total_weight = self.class_counts.sum().float()

        for i in range(self.num_classes):
            class_accuracy = self.correct[i] / self.class_counts[i] if self.class_counts[i] > 0 else 0
            weighted_accuracy += class_accuracy * (self.class_counts[i] / total_weight)

        return weighted_accuracy