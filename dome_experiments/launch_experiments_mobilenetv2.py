#!/usr/bin/env python3
"""
Generic DP training with AdamCorr + MobileNetV2 + GroupNorm (DP-friendly).

Supported datasets:
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
import csv
import hashlib
import os
import random
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus_local import PrivacyEngine
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
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

# Standard ImageNet normalization (good default for TinyImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------------------
# TinyImageNet auto-download
# ------------------------------------------------------------------------

TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINYIMAGENET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"

def log_training_loss(step: int, loss: float, csv_path: str, remove_dom: bool, seed: int):
    """
    Append training loss at a given step to a CSV file.
    """
    write_header = not os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "loss", "remove_dom", "seed"])
        writer.writerow([step, loss, remove_dom, seed])
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
    # Accept either parent dir or exact folder.
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
        return  # already prepared

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
        "in_channels": 1,
        "num_classes": 10,
        "default_csv": "../results/training_metrics_mnist_mnv2.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../mnist",
        "sketched": {"seeds": [5, 6, 7]},
        "debias_sketched": True,
    },
    "fashionmnist": {
        "in_channels": 1,
        "num_classes": 10,
        "default_csv": "../results/training_metrics_fashionmnist_mnv2.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../fashionmnist",
        "sketched": {"seeds": [2, 3, 4, 5, 6]},
        "debias_sketched": True,
    },
    "cifar10": {
        "in_channels": 3,
        "num_classes": 10,
        "default_csv": "../results/training_metrics_cifar10_mnv2.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../cifar10",
        "sketched": {"seeds": [2, 3, 4, 5, 6]},
        "debias_sketched": True,
    },
    "tinyimagenet": {
        "in_channels": 3,
        "num_classes": 200,
        "default_csv": "../results/training_metrics_tinyimagenet_mnv2.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../tinyimagenet",
        "sketched": {"seeds": [2, 3, 4,5,6]},
        "debias_sketched": True,
    },
}


# ------------------------------------------------------------------------
# MobileNetV2 with GroupNorm (replace BatchNorm2d)
# ------------------------------------------------------------------------

def _pick_gn_groups(channels: int, preferred: int = 8) -> int:
    for g in [preferred, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def replace_bn_with_gn(module: nn.Module, preferred_groups: int = 8) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = _pick_gn_groups(c, preferred_groups)
            setattr(module, name, nn.GroupNorm(g, c))
        else:
            replace_bn_with_gn(child, preferred_groups)
    return module


class MobileNetV2GN(nn.Module):
    def __init__(self, num_classes: int, gn_groups: int = 8, width_mult: float = 1.0):
        super().__init__()
        base = mobilenet_v2(weights=None, width_mult=width_mult)
        base = replace_bn_with_gn(base, preferred_groups=gn_groups)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

    def name(self):
        return f"MobileNetV2GN_w{self.model.features[0][0].out_channels}"


# ------------------------------------------------------------------------
# Training / eval
# ------------------------------------------------------------------------

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch, remove_dom=True, seed=0):
    model.train()
    criterion = nn.CrossEntropyLoss()

    start_step = (epoch - 1) * len(train_loader)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        _step = start_step + batch_idx
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # AdamCorr is expected to return extra debug scalars
        optimizer.step()
        # # Log loss
        log_training_loss(
            step=_step,
            loss=loss.item(),
            csv_path="training_loss_tinyimagenet.csv",
            remove_dom = remove_dom,
            seed= seed
        )


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
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
        # Convert to 3ch so MobileNetV2 can be used directly
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
        num_workers=8,
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
    csv_filename = args.csv_filename or cfg["default_csv"]
    file_exists = os.path.isfile(csv_filename)

    seeds = cfg["sketched"]["seeds"]
    debias = cfg["debias_sketched"]

    for seed in seeds:
        set_seed(seed)

        for remove_dom in [True, False]:
            try:
                model = MobileNetV2GN(
                    num_classes=num_classes,
                    gn_groups=args.gn_groups,
                    width_mult=args.width_mult,
                ).to(device)

                nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
                    train(args, model, device, train_loader_p, optimizer, privacy_engine, epoch, remove_dom=remove_dom, seed=seed)
                    test_accuracy = test(model, device, test_loader)

                    row = {
                        "dataset": ds,
                        "epoch": epoch,
                        "batch_size": args.batch_size,
                        "accuracy": test_accuracy,
                        "compression_rate": args.compression_rate,
                        "debias": debias,
                        "remove_dom": remove_dom,
                        "seed": seed,
                        "nb_dims_pca": args.nb_dims_pca,
                        "arch": f"MobileNetV2GN_w{args.width_mult}",
                    }
                    pd.DataFrame([row]).to_csv(csv_filename, mode="a", header=not file_exists, index=False)
                    file_exists = True

            except Exception as error:
                print("An exception occurred:", error)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DP training with AdamCorr + MobileNetV2GN",
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
    parser.add_argument("-n", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cuda:0, cpu)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader num_workers")

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
    parser.add_argument("--csv-filename", type=str, default=None, help="Optional CSV filename override")

    # MobileNetV2 knobs
    parser.add_argument("--width-mult", type=float, default=0.5, help="MobileNetV2 width multiplier")
    parser.add_argument("--gn-groups", type=int, default=8, help="Preferred GN groups (auto-adjusted per layer)")

    # AdamCorr knobs
    parser.add_argument("--adamcorr-lr", type=float, default=1e-3, help="AdamCorr learning rate")
    parser.add_argument("--nb-dims-pca", type=int, default=400, help="nb_dims_pca")
    parser.add_argument("--compression-rate", type=int, default=1, help="compression_rate")

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
