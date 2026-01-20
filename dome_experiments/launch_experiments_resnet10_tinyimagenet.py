#!/usr/bin/env python3
"""
Launcher: DP training with AdamCorr + ResNet-8 (Opacus-compatible) + GroupNorm.

Refactor summary (requested):
- Training loss is logged to a dataset-specific CSV from DATASET_CONFIG.
- Gradient-subspace fractions are logged to a dataset-specific CSV from DATASET_CONFIG,
  with full run metadata (dataset/arch/seed/remove_dom/etc.) and injected into the optimizer
  via optimizer.set_logging(...).
- Accuracy metrics are logged as before to default_csv (or --csv-filename override).
- All CSV rows explicitly include remove_dom, arch, seed, etc.

Datasets:
  - MNIST
  - FashionMNIST
  - CIFAR-10
  - TinyImageNet (tiny-imagenet-200)

TinyImageNet auto-download:
  - Downloads the official CS231n zip if missing
  - Unzips under --data-root
  - Builds an ImageFolder-compatible validation folder automatically

Expected TinyImageNet layout after unzip:
  <data_root>/tiny-imagenet-200/train/<wnid>/images/*.JPEG
  <data_root>/tiny-imagenet-200/val/images/*.JPEG
  <data_root>/tiny-imagenet-200/val/val_annotations.txt
"""

import argparse
import hashlib
import os
import random
import shutil
import urllib.request
import zipfile
import csv
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus_local import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

from opacus_local.optimizers.DOME_optim import AdamCorr


# ------------------------------------------------------------------------
# Dataset stats
# ------------------------------------------------------------------------

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------
# TinyImageNet auto-download
# ------------------------------------------------------------------------

TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINYIMAGENET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"


def _md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_and_prepare_tinyimagenet(data_root: str) -> str:
    """
    Ensures tiny-imagenet-200 is present under data_root (or data_root itself is that folder).
    Downloads and unzips if missing.

    Returns: path to tiny-imagenet-200 directory.
    """
    ti_root = data_root
    if os.path.basename(os.path.normpath(ti_root)) != "tiny-imagenet-200":
        ti_root = os.path.join(data_root, "tiny-imagenet-200", "tiny-imagenet-200")

    if os.path.isdir(ti_root) and os.path.isdir(os.path.join(ti_root, "train")):
        return ti_root

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")

    if not os.path.isfile(zip_path):
        print(f"[TinyImageNet] Downloading from {TINYIMAGENET_URL} -> {zip_path}")
        urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path)

    md5 = _md5(zip_path)
    if md5 != TINYIMAGENET_MD5:
        raise RuntimeError(
            f"[TinyImageNet] MD5 mismatch for {zip_path}: got {md5}, expected {TINYIMAGENET_MD5}. "
            f"Delete the zip and retry."
        )

    print(f"[TinyImageNet] Unzipping {zip_path} -> {data_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    if not os.path.isdir(ti_root) or not os.path.isdir(os.path.join(ti_root, "train")):
        raise RuntimeError(f"[TinyImageNet] Unzip finished but expected folder not found: {ti_root}")

    return ti_root


def prepare_tinyimagenet_val(val_dir: str, out_dir: str):
    """
    Create ImageFolder-compatible validation directory:
      out_dir/<wnid>/*.JPEG
    based on val_annotations.txt.
    """
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    ann_path = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")

    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing TinyImageNet val annotations: {ann_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing TinyImageNet val images dir: {images_dir}")

    mapping = {}
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    for img_name, wnid in mapping.items():
        src = os.path.join(images_dir, img_name)
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.join(out_dir, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, img_name)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)


# ------------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------------

DATASET_CONFIG = {
    "mnist": {
        "num_classes": 10,
        "default_csv": "../results/training_metrics_mnist_resnet8_gn.csv",
        "default_loss_csv": "../results/training_loss_mnist_resnet8_gn.csv",
        "default_fractions_csv": "../results/gradient_fractions_mnist_resnet8_gn.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../mnist",
        "sketched": {"seeds": [5, 6, 7]},
        "debias_sketched": True,
    },
    "fashionmnist": {
        "num_classes": 10,
        "default_csv": "../results/training_metrics_fashionmnist_resnet8_gn.csv",
        "default_loss_csv": "../results/training_loss_fashionmnist_resnet8_gn.csv",
        "default_fractions_csv": "../results/gradient_fractions_fashionmnist_resnet8_gn.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../fashionmnist",
        "sketched": {"seeds": [2, 3, 4, 5, 6]},
        "debias_sketched": True,
    },
    "cifar10": {
        "num_classes": 10,
        "default_csv": "../results/training_metrics_cifar10_resnet8_gn.csv",
        "default_loss_csv": "../results/training_loss_cifar10_resnet8_gn.csv",
        "default_fractions_csv": "../results/gradient_fractions_cifar10_resnet8_gn.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../cifar10",
        "sketched": {"seeds": [2, 3, 4, 5, 6]},
        "debias_sketched": True,
    },
    "tinyimagenet": {
        "num_classes": 200,
        "default_csv": "../results/training_metrics_tinyimagenet_resnet8_gn.csv",
        "default_loss_csv": "../results/training_loss_tinyimagenet_resnet8_gn.csv",
        "default_fractions_csv": "../results/gradient_fractions_tinyimagenet_resnet8_gn.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../tinyimagenet",
        "sketched": {"seeds": [2, 3, 4, 5, 6]},
        "debias_sketched": True,
    },
}


# ------------------------------------------------------------------------
# Logging helpers (loss)
# ------------------------------------------------------------------------

LOSS_FIELDS = [
    "dataset",
    "arch",
    "seed",
    "remove_dom",
    "debias",
    "compression_rate",
    "nb_dims_pca",
    "disable_dp",
    "sigma",
    "max_per_sample_grad_norm",
    "epoch",
    "step",
    "loss",
]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_row_csv(csv_path: str, fieldnames: List[str], row: Dict) -> None:
    _ensure_parent_dir(csv_path)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_training_loss(
    *,
    csv_path: str,
    run_meta: Dict,
    epoch: int,
    step: int,
    loss: float,
) -> None:
    row = dict(run_meta)
    row.update({"epoch": int(epoch), "step": int(step), "loss": float(loss)})
    _append_row_csv(csv_path, LOSS_FIELDS, row)


# ------------------------------------------------------------------------
# GroupNorm helpers
# ------------------------------------------------------------------------

def _pick_gn_groups(channels: int, preferred: int = 8) -> int:
    for g in [preferred, 32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def gn(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    g = _pick_gn_groups(num_channels, preferred_groups)
    return nn.GroupNorm(g, num_channels)


# ------------------------------------------------------------------------
# ResNet-8 (CIFAR-style) with GroupNorm (Opacus-friendly)
# ------------------------------------------------------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, gn_groups: int = 8):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.gn1 = gn(planes, preferred_groups=gn_groups)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.gn2 = gn(planes, preferred_groups=gn_groups)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                gn(planes, preferred_groups=gn_groups),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet8GN(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16, gn_groups: int = 8):
        super().__init__()
        self.in_planes = width
        self.gn_groups = gn_groups

        self.conv1 = conv3x3(3, width, stride=1)
        self.gn1 = gn(width, preferred_groups=gn_groups)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(width, blocks=1, stride=1)
        self.layer2 = self._make_layer(width * 2, blocks=1, stride=2)
        self.layer3 = self._make_layer(width * 4, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [BasicBlockGN(self.in_planes, planes, stride=stride, gn_groups=self.gn_groups)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockGN(self.in_planes, planes, stride=1, gn_groups=self.gn_groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def name(self):
        return f"ResNet8GN_w{self.in_planes}_g{self.gn_groups}"


# ------------------------------------------------------------------------
# Training / eval
# ------------------------------------------------------------------------

def train_one_epoch(
    *,
    args,
    model,
    device,
    train_loader,
    optimizer,
    epoch: int,
    loss_csv_path: str,
    run_meta: Dict,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    start_step = (epoch - 1) * len(train_loader)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        step = start_step + batch_idx
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        log_training_loss(
            csv_path=loss_csv_path,
            run_meta=run_meta,
            epoch=epoch,
            step=step,
            loss=loss.item(),
        )


@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= max(1, len(test_loader))
    total = len(test_loader.dataset)
    acc = 100.0 * correct / total
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)\n")
    return acc / 100.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------------

def build_dataloaders(args):
    ds = args.dataset.lower()

    if ds == "mnist":
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,) * 3, (MNIST_STD,) * 3),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,) * 3, (MNIST_STD,) * 3),
        ])
        train_dataset = datasets.MNIST(args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=test_transform)

    elif ds == "fashionmnist":
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,) * 3, (FASHION_MNIST_STD,) * 3),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,) * 3, (FASHION_MNIST_STD,) * 3),
        ])
        train_dataset = datasets.FashionMNIST(args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(args.data_root, train=False, download=True, transform=test_transform)

    elif ds == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        train_dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)

    elif ds == "tinyimagenet":
        ti_root = download_and_prepare_tinyimagenet(args.data_root)
        train_dir = os.path.join(ti_root, "train")
        val_dir = os.path.join(ti_root, "val")

        val_out = os.path.join(ti_root, "val_images_by_class")
        prepare_tinyimagenet_val(val_dir=val_dir, out_dir=val_out)

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=val_out, transform=test_transform)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader


# ------------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------------

def run_experiments(args, device, train_loader, test_loader):
    ds = args.dataset.lower()
    cfg = DATASET_CONFIG[ds]

    num_classes = cfg["num_classes"]

    metrics_csv = args.csv_filename or cfg["default_csv"]
    loss_csv = cfg["default_loss_csv"]
    fractions_csv = cfg["default_fractions_csv"]

    metrics_exists = os.path.isfile(metrics_csv)

    seeds = cfg["sketched"]["seeds"]
    debias = cfg["debias_sketched"]

    for seed in seeds:
        set_seed(seed)

        for remove_dom in [False, True]:
            try:
                model = ResNet8GN(
                    num_classes=num_classes,
                    width=args.resnet_width,
                    gn_groups=args.gn_groups,
                ).to(device)

                nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                arch_str = f"ResNet8GN_w{args.resnet_width}_g{args.gn_groups}"

                optimizer = AdamCorr(
                    model.parameters(),
                    lr=args.adamcorr_lr,
                    dp_batch_size=args.batch_size,
                    dp_noise_multiplier=1,
                    dp_l2_norm_clip=1,
                    eps_root=1e-10,
                    numel=nb_params,
                    use_sketching=True,
                    debias=debias,
                    compression_rate=args.compression_rate,
                    use_momentum=False,
                    betas=(0.9, 0.999),
                    remove_dom=remove_dom,
                    nb_dims_pca=args.nb_dims_pca,
                    seed=seed,
                )

                # Shared run metadata used in BOTH loss and fraction logs (and also in metrics rows)
                run_meta = {
                    "dataset": ds,
                    "arch": arch_str,
                    "seed": int(seed),
                    "remove_dom": bool(remove_dom),
                    "debias": bool(debias),
                    "compression_rate": int(args.compression_rate),
                    "nb_dims_pca": int(args.nb_dims_pca),
                    "disable_dp": bool(args.disable_dp),
                    "sigma": float(args.sigma),
                    "max_per_sample_grad_norm": float(args.max_per_sample_grad_norm),
                }

                # Inject optimizer-side logging config (fractions)
                optimizer.set_logging(
                    fractions_csv_path=fractions_csv,
                    run_meta=run_meta,
                    log_fractions_every=args.log_fractions_every,
                    enable_fraction_logging=not args.disable_fraction_logging,
                )

                privacy_engine = None
                if not args.disable_dp:
                    privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
                    model, optimizer, train_loader_p = privacy_engine.make_private(
                        module=model,
                        clipping="dome",
                        optimizer=optimizer,
                        poisson_sampling=False,
                        data_loader=train_loader,
                        noise_multiplier=args.sigma,
                        max_grad_norm=args.max_per_sample_grad_norm,
                    )
                else:
                    train_loader_p = train_loader

                for epoch in range(1, args.epochs + 1):
                    train_one_epoch(
                        args=args,
                        model=model,
                        device=device,
                        train_loader=train_loader_p,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss_csv_path=loss_csv,
                        run_meta=run_meta,
                    )
                    test_accuracy = test(model, device, test_loader)

                    row = {
                        "dataset": ds,
                        "epoch": epoch,
                        "batch_size": args.batch_size,
                        "accuracy": float(test_accuracy),
                        "compression_rate": int(args.compression_rate),
                        "debias": bool(debias),
                        "remove_dom": bool(remove_dom),
                        "seed": int(seed),
                        "nb_dims_pca": int(args.nb_dims_pca),
                        "arch": arch_str,
                        "disable_dp": bool(args.disable_dp),
                        "sigma": float(args.sigma),
                        "max_per_sample_grad_norm": float(args.max_per_sample_grad_norm),
                    }
                    pd.DataFrame([row]).to_csv(metrics_csv, mode="a", header=not metrics_exists, index=False)
                    metrics_exists = True

            except Exception as error:
                print("An exception occurred:", error)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DP training with AdamCorr + ResNet-8 + GroupNorm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "fashionmnist", "cifar10", "tinyimagenet"],
        help="Dataset to use",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="Test batch size")
    parser.add_argument("-n", "--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cuda:0, cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")

    # DP params
    parser.add_argument("--disable-dp", action="store_true", default=False, help="Disable DP training")
    parser.add_argument("--secure-rng", action="store_true", default=False, help="Enable secure RNG (slower)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier")
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        dest="max_per_sample_grad_norm",
        type=float,
        default=1.0,
        help="Clip per-sample gradients to this norm",
    )

    # Data / logging
    parser.add_argument("--data-root", type=str, default="../data", help="Dataset root")
    parser.add_argument("--csv-filename", type=str, default=None, help="Optional metrics CSV filename override")

    # ResNet knobs
    parser.add_argument("--resnet-width", type=int, default=16, help="Base width for ResNet-8 (channels)")
    parser.add_argument("--gn-groups", type=int, default=8, help="Preferred GN groups")

    # AdamCorr knobs
    parser.add_argument("--adamcorr-lr", type=float, default=1e-3, help="AdamCorr learning rate")
    parser.add_argument("--nb-dims-pca", type=int, default=200, help="nb_dims_pca")
    parser.add_argument("--compression-rate", type=int, default=1, help="compression_rate")

    # Fractions logging knobs
    parser.add_argument(
        "--log-fractions-every",
        type=int,
        default=1,
        help="Log gradient subspace fractions every N optimizer steps (reduces IO).",
    )
    parser.add_argument(
        "--disable-fraction-logging",
        action="store_true",
        default=False,
        help="Disable optimizer-side fraction logging entirely.",
    )

    args = parser.parse_args()

    ds = args.dataset.lower()
    if ds not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cfg = DATASET_CONFIG[ds]
    torch.set_default_dtype(cfg["default_dtype"])

    # If user keeps default "../data", use dataset-specific default root
    if args.data_root == "../data":
        args.data_root = cfg["default_data_root"]

    device = torch.device(args.device)
    train_loader, test_loader = build_dataloaders(args)
    run_experiments(args, device, train_loader, test_loader)


if __name__ == "__main__":
    main()
