"""多模态分类训练入口。

默认直接读取根目录 config.py 中的 DEFAULT_CONFIG。也可以通过
--config 指定 json/yaml 配置文件覆盖默认参数。

示例：
    python train.py
    python train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
g
from mtf.datatset.lowdim_loader import MultiModalDataset
from mtf.model.model import MultiModalClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="训练时序 + 图像多模态分类器")
    parser.add_argument("--config", default=None, help="json/yaml 配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖配置中的学习率")
    parser.add_argument("--device", default=None, help="覆盖配置中的设备: auto/cpu/cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def stratified_split(dataset: MultiModalDataset, val_ratio: float, seed: int):
    """按类别做 train/val 切分，避免小数据集随机切分后类别不均衡。"""
    label_to_indices = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        label_to_indices[sample["label"]].append(index)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for indices in label_to_indices.values():
        indices = indices[:]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_loaders(config: dict):
    data_config = config["data"]
    train_config = config["train"]

    dataset = MultiModalDataset(data_config)
    train_dataset, val_dataset = stratified_split(
        dataset=dataset,
        val_ratio=float(data_config["val_ratio"]),
        seed=int(config["seed"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for label, pred in zip(labels.cpu(), preds.cpu()):
        confusion[int(label), int(pred)] += 1

    precisions = []
    recalls = []
    f1_scores = []
    for class_index in range(num_classes):
        tp = confusion[class_index, class_index].item()
        fp = confusion[:, class_index].sum().item() - tp
        fn = confusion[class_index, :].sum().item() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": correct / max(total, 1),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion": confusion,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        signal = batch["signal"].to(device)
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        logits = model(signal, image)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, model.config["model"]["num_classes"])
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        signal = batch["signal"].to(device)
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        logits = model(signal, image)
        loss = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, model.config["model"]["num_classes"])
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def save_checkpoint(save_dir: Path, model, optimizer, epoch: int, config: dict, metrics: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key != "confusion"
        },
    }
    torch.save(checkpoint, save_dir / "best.pt")
    (save_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def deep_update(base: dict, override: dict) -> dict:
    """递归合并配置，override 中的值会覆盖 base。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> dict:
    """读取配置文件并和默认配置合并。

    支持 JSON；如果环境安装了 PyYAML，也支持 YAML/YML。
    不传 config_path 时直接返回 DEFAULT_CONFIG 的深拷贝。
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return config

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        override = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("读取 YAML 配置需要安装 PyYAML。") from exc
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError("配置文件只支持 .json / .yaml / .yml")

    return deep_update(config, override)

def main():
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.device is not None:
        config["train"]["device"] = args.device

    set_seed(int(config["seed"]))
    device = resolve_device(config["train"]["device"])

    train_loader, val_loader = build_loaders(config)
    model = MultiModalClassifier(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    save_dir = Path(config["train"]["save_dir"]).expanduser().resolve()
    best_f1 = -1.0

    print(f"device: {device}")
    print(f"train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(save_dir, model, optimizer, epoch, config, val_metrics)

        if epoch % int(config["train"]["print_every"]) == 0:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f}"
            )

    print("best macro_f1:", best_f1)
    print("checkpoint:", save_dir / "best.pt")


if __name__ == "__main__":
    main()
