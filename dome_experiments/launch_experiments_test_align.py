#!/usr/bin/env python3

"""
Generic DP training with AdamCorr and ResNet-8 + GroupNorm.

Supported datasets:
  - MNIST
  - FashionMNIST
  - CIFAR-10
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


# ------------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------------

DATASET_CONFIG = {
    "mnist": {
        "in_channels": 1,
        "num_classes": 10,
        "default_csv": "../results/training_metrics_mnist_compression.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../mnist",
        # experiment grids
        "sketched": {
            "seeds": [2, 4, 5, 6, 7],
            "dp_scales": [0],
            "compression_rates": [10,50,100,500, 1000,5000,10000],
        },
        "baseline": {
            "dp_scales": [0, 0.1, 0.2,0.5,1.0],
            "baseline_rank_const": 26000 // 1000,
        },
        "use_cifar_extras": False,   # no betas/use_adam_preconditioning/accumulate
        "debias_sketched": True,
        "debias_baseline": False,
    },
    "fashionmnist": {
        "in_channels": 1,
        "num_classes": 10,
        "default_csv": "../results/training_metrics_fashionmnist_generic_test.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../fashionmnist",
        "sketched": {
            "seeds": [2, 3, 4,5,6],
            "dp_scales": [0.0],
            "compression_rates": [1, 2, 5, 10,20,50,100, 200, 500, 1000],
        },
        "baseline": {
            "dp_scales": [0, 0.1, 0.2,0.5,1.0],
            "baseline_rank_const": 26000 // 1000,
        },
        "use_cifar_extras": False,
        "debias_sketched": True,
        "debias_baseline": False,
    },
    "cifar10": {
        "in_channels": 3,
        "num_classes": 10,
        "default_csv": "../results/training_metrics_cifar10_compression.csv",
        "default_dtype": torch.float32,
        "default_data_root": "../cifar10",
        "sketched": {
            "seeds": [3,4,5,6],
            "dp_scales": [0.0],
            "compression_rates": [10,50,100,500, 1000,5000,10000],
        },
        "baseline": {
            "dp_scales": [0, 0.1, 0.2,0.5,1.0],
            "baseline_rank_const": 26000 // 1000,
        },
        "use_cifar_extras": True,   # betas/use_adam_preconditioning/accumulate/seed
        "debias_sketched": True,
        "debias_baseline": False,
    },
}


# ------------------------------------------------------------------------
# Model: generic ResNet-8 + GroupNorm
# ------------------------------------------------------------------------

class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=8):
        super().__init__()
        g1 = num_groups if (planes % num_groups) == 0 else 1
        g2 = num_groups if (planes % num_groups) == 0 else 1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(g1, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(g2, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(num_groups if (planes % num_groups) == 0 else 1, planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet8GN(nn.Module):
    """
    Small ResNet-8:
      - conv1
      - 3 stages with 1 BasicBlockGN each
      => 1 + 2*3 = 7 conv layers + fc = 8 "layers"

    Works for MNIST / FashionMNIST (in_channels=1) and CIFAR-10 (in_channels=3).
    """

    def __init__(self, in_channels=1, num_classes=10, num_groups=8):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups if (16 % num_groups) == 0 else 1, 16)

        self.layer1 = self._make_layer(
            BasicBlockGN, 16, num_blocks=1, stride=1, num_groups=num_groups
        )
        self.layer2 = self._make_layer(
            BasicBlockGN, 32, num_blocks=1, stride=2, num_groups=num_groups
        )
        self.layer3 = self._make_layer(
            BasicBlockGN, 64, num_blocks=1, stride=2, num_groups=num_groups
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlockGN.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, num_groups):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, num_groups=num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def name(self):
        return "ResNet8GN"


# ------------------------------------------------------------------------
# Training / eval helpers
# ------------------------------------------------------------------------

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    all_adam_precond_norms = []
    all_custom_precond_norms = []
    all_mean_custom_norms = []
    all_regular_norms = []
    all_mean_norms = []

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        custom_precond_norms, adam_precond_norms, mean_customnorms, regular_norms, mean_norm = optimizer.step()
        all_adam_precond_norms.append(adam_precond_norms)
        all_custom_precond_norms.append(custom_precond_norms)
        all_mean_custom_norms.append(mean_customnorms)
        all_regular_norms.append(regular_norms)
        all_mean_norms.append(mean_norm)
        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    return (
        all_adam_precond_norms,
        all_custom_precond_norms,
        all_mean_custom_norms,
        all_regular_norms,
        all_mean_norms,
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
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )
        train_dataset = datasets.MNIST(
            args.data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            args.data_root, train=False, download=True, transform=transform
        )

    elif ds == "fashionmnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
            ]
        )
        train_dataset = datasets.FashionMNIST(
            args.data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            args.data_root, train=False, download=True, transform=transform
        )

    elif ds == "cifar10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        train_dataset = datasets.CIFAR10(
            root=args.data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


# ------------------------------------------------------------------------
# Unified experiment runner
# ------------------------------------------------------------------------

def run_experiments(args, device, train_loader, test_loader):
    ds = args.dataset.lower()
    cfg = DATASET_CONFIG[ds]

    in_channels = cfg["in_channels"]
    num_classes = cfg["num_classes"]
    default_csv = cfg["default_csv"]

    sk_cfg = cfg["sketched"]
    bl_cfg = cfg["baseline"]

    sketched_seeds = sk_cfg["seeds"]
    sketched_dp_scales = sk_cfg["dp_scales"]
    sketched_compression_rates = sk_cfg["compression_rates"]

    baseline_dp_scales = bl_cfg["dp_scales"]
    baseline_rank_const = bl_cfg["baseline_rank_const"]

    debias_sketched = cfg["debias_sketched"]
    debias_baseline = cfg["debias_baseline"]

    csv_filename = args.csv_filename or default_csv
    file_exists = os.path.isfile(csv_filename)

    # -------------------- Sketched runs --------------------
    debias = debias_sketched
    for seed in sketched_seeds:
        set_seed(seed)
        for compression_rate in sketched_compression_rates:
            for remove_dom in [True, False]:
                # for remove_mean in [True, False]:
                try:
                    model = ResNet8GN(
                        in_channels=in_channels,
                        num_classes=num_classes,
                    ).to(device)
                    use_sketching = True

                    nb_params = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )

                    adamcorr_kwargs = dict(
                        lr=0.001,
                        dp_batch_size=args.batch_size,
                        dp_noise_multiplier=1,
                        dp_l2_norm_clip=1,
                        eps_root=1e-10,
                        numel=nb_params,
                        use_sketching=use_sketching,
                        debias=debias,
                        compression_rate=compression_rate,
                        use_momentum = False,
                        betas=(0.9,0.999),
                        remove_dom=remove_dom
                    )
                    optimizer = AdamCorr(
                        model.parameters(),
                        **adamcorr_kwargs,
                        # remove_mean=remove_mean
                    )

                    privacy_engine = None
                    if not args.disable_dp:
                        privacy_engine = PrivacyEngine(
                            secure_mode=args.secure_rng
                        )
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
                        (
                            custom,
                            adam,
                            mean_custom_grads_norms,
                            regular_norms,
                            mean_norm,
                        ) = train(
                            args,
                            model,
                            device,
                            train_loader_p,
                            optimizer,
                            privacy_engine,
                            epoch,
                        )
                        test_accuracy = test(model, device, test_loader)
                        data = [
                            {
                                "dataset": ds,
                                "epoch": epoch,
                                "batch_size": args.batch_size,
                                "accuracy": test_accuracy,
                                "compression_rate": compression_rate,
                                "debias": debias,
                                "remove_dom": remove_dom,
                                "seed": seed,
                                # "remove_mean": remove_mean
                            }
                        ]
                        df_epoch = pd.DataFrame(data)
                        df_epoch.to_csv(
                            csv_filename,
                            mode="a",
                            header=not file_exists,
                            index=False,
                        )
                        file_exists = True
                except Exception as error:
                                    print("An exception occurred ", error)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic Opacus Example (ResNet-8 + AdamCorr)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "fashionmnist", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla optimizer",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="Where dataset is/will be stored",
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default=None,
        help="Optional CSV filename override",
    )

    args = parser.parse_args()
    ds = args.dataset.lower()
    if ds not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cfg = DATASET_CONFIG[ds]

    # Dataset-specific default dtype + default data root
    torch.set_default_dtype(cfg["default_dtype"])
    if args.data_root == "../data":
        args.data_root = cfg["default_data_root"]

    device = torch.device(args.device)
    train_loader, test_loader = build_dataloaders(args)

    run_experiments(args, device, train_loader, test_loader)


if __name__ == "__main__":
    main()
