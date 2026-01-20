#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Runs CIFAR-10 training with differential privacy using a CIFAR-style ResNet-8.
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

# CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

torch.set_default_dtype(torch.float32)

# ---------------------- DP-safe ResNet-18 with GroupNorm ----------------------

def _gn(groups, num_channels):
    return nn.GroupNorm(groups if (num_channels % groups) == 0 else 1, num_channels)
# ---------------- DP-safe ResNet-8 (GroupNorm) ----------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # GroupNorm: channels must be divisible by groups; fall back to 1 group if needed
        g1 = gn_groups if (planes % gn_groups) == 0 else 1
        self.gn1 = nn.GroupNorm(g1, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        g2 = gn_groups if (planes % gn_groups) == 0 else 1
        self.gn2 = nn.GroupNorm(g2, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(gn_groups if (planes % gn_groups) == 0 else 1, planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet8CIFAR(nn.Module):
    """
    CIFAR-style ResNet with depth 6n+2 = 8 -> n=1 (one BasicBlock per stage).
    Uses GroupNorm everywhere (DP-safe).
    """
    def __init__(self, num_classes=10, gn_groups=8):
        super().__init__()
        self.in_planes = 16
        self.gn_groups = gn_groups

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.gn0 = nn.GroupNorm(gn_groups if (16 % gn_groups) == 0 else 1, 16)

        self.layer1 = self._make_layer(16, 1, stride=1)
        self.layer2 = self._make_layer(32, 1, stride=2)
        self.layer3 = self._make_layer(64, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for st in strides:
            layers.append(BasicBlock(self.in_planes, planes, st, gn_groups=self.gn_groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn0(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def name(self):
        return "ResNet8CIFAR_GN"
# --------------------------------------------------------------

# --------------------------------------------------------------


# -----------------------------------------------------------------------------------------

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    all_adam_precond_norms = []
    all_custom_precond_norms = []
    all_mean_custom_norms = []
    all_regular_norms = []
    all_mean_norms = []

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=47,
        T_mult=1,
    )

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        custom_precond_norms, adam_precond_norms, mean_customnorms, regular_norms, mean_norm = optimizer.step()
        # scheduler.step()
        all_adam_precond_norms.append(adam_precond_norms)
        all_custom_precond_norms.append(custom_precond_norms)
        all_mean_custom_norms.append(mean_customnorms)
        all_regular_norms.append(regular_norms)
        all_mean_norms.append(mean_norm)
        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    return all_adam_precond_norms, all_custom_precond_norms, all_mean_custom_norms, all_regular_norms, all_mean_norms


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


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus CIFAR-10 Example (ResNet-8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, metavar="B", help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024, metavar="TB", help="input batch size for testing")
    parser.add_argument("-n", "--epochs", type=int, default=20, metavar="N", help="number of epochs to train")
    parser.add_argument("-r", "--n-runs", type=int, default=1, metavar="R", help="number of runs to average on")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate")
    parser.add_argument("--sigma", type=float, default=1.0, metavar="S", help="Noise multiplier")
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C",
                        help="Clip per-sample gradients to this norm")
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta")
    parser.add_argument("--device", type=str, default="cuda", help="GPU ID for this process")
    parser.add_argument("--save-model", action="store_true", default=False, help="Save the trained model")
    parser.add_argument("--disable-dp", action="store_true", default=False,
                        help="Disable privacy training and just train with vanilla optimizer")
    parser.add_argument("--secure-rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost")
    parser.add_argument("--data-root", type=str, default="../cifar10", help="Where CIFAR-10 is/will be stored")
    args = parser.parse_args()
    device = torch.device(args.device)

    # ------------------------------- CIFAR-10 loaders -------------------------------
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform),
        batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    csv_filename = "training_metrics_cifar10_resnet8_2711_b16.csv"
    file_exists = os.path.isfile(csv_filename)

    debias = True  # keep your toggles consistent with original
    for seed in [2,3,4]:
        set_seed(seed)
        for dp_scale in [0,0.1,0.2,0.3]:
        # for dp_scale in [1]:
            for compression_rate in [10, 50, 200, 500,1000]:
            # for compression_rate in [50]:
            # for compression_rate in [200,500]:
                # for use_random_sketch in [False, True]:
                for use_random_sketch in [False, True]:
                    # juju was there <3
                    numertrick_list = [False]
                    for use_numertrick in numertrick_list:
                        annihil= False if use_random_sketch else True
                        use_preconditioning_list = [False]
                        for use_preconditioning in use_preconditioning_list:
                            for _ in range(args.n_runs):
                                try:
                                    model = ResNet8CIFAR(num_classes=10).to(device)
                                    use_sketching = True

                                    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                                    rank = max(1, nb_params // compression_rate)

                                    optimizer = AdamCorr(model.parameters(), lr=0.001, betas= (0.9,0.999),dp_batch_size=128, dp_noise_multiplier=1,
                                                         dp_l2_norm_clip=1, eps_root=1e-10, numel=nb_params, rank=rank,
                                                         random_sketch=use_random_sketch, use_sketching=use_sketching,
                                                         use_preconditioning=use_preconditioning, debias=debias,
                                                         dp_scale=dp_scale, use_numertrick=use_numertrick, annihil=annihil, use_adam_preconditioning=True, seed=seed, accumulate=True
                                    )

                                    privacy_engine = None
                                    if not args.disable_dp:
                                        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
                                        model, optimizer, train_loader = privacy_engine.make_private(
                                            module=model,
                                            clipping="dome",
                                            optimizer=optimizer,
                                            poisson_sampling=False,
                                            data_loader=train_loader,
                                            noise_multiplier=args.sigma,
                                            max_grad_norm=args.max_per_sample_grad_norm,
                                        )

                                    for epoch in range(1, args.epochs + 1):
                                        custom, adam, mean_custom_grads_norms, regular_norms, mean_norm = train(
                                            args, model, device, train_loader, optimizer, privacy_engine, epoch
                                        )
                                        test_accuracy = test(model, device, test_loader)
                                        data = [{
                                            'epoch': epoch,
                                            'use_random_sketching': use_random_sketch,
                                            'accuracy': test_accuracy,
                                            'compression_rate': compression_rate,
                                            'use_sketching': use_sketching,
                                            'dp_scale': dp_scale,
                                            'use_preconditioning': use_preconditioning,
                                            'debias': debias,
                                            'use_numertrick': use_numertrick,
                                            'annihil': annihil,
                                            'seed': seed
                                        }]
                                        df_epoch = pd.DataFrame(data)
                                        df_epoch.to_csv(
                                            csv_filename,
                                            mode='a',
                                            header=not file_exists,
                                            index=False
                                        )
                                        file_exists = True
                                except Exception as error:
                                    print("An exception occurred ", error)

    # Non-sketched runs (baseline)
    debias = False
    for seed in [2, 3, 4]:
        set_seed(seed)
        for dp_scale in [0]:
            for _ in range(args.n_runs):
                compression_rate = 100
                use_random_sketch = False
                use_sketching = False
                use_preconditioning = False

                model = ResNet8CIFAR(num_classes=10).to(device)
                nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                rank = max(1, 26000 // 1000)

                optimizer = AdamCorr(
                    model.parameters(),
                    lr=args.lr,
                    dp_batch_size=args.batch_size,
                    dp_noise_multiplier=1,
                    dp_l2_norm_clip=1,
                    eps_root=1e-8,
                    numel=nb_params,
                    rank=rank,
                    random_sketch=use_random_sketch,
                    use_sketching=use_sketching,
                    use_preconditioning=use_preconditioning,
                    debias=debias,
                    dp_scale=dp_scale
                )

                privacy_engine = None
                if not args.disable_dp:
                    privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
                    model, optimizer, train_loader = privacy_engine.make_private(
                        module=model,
                        clipping="dome",
                        optimizer=optimizer,
                        poisson_sampling=False,
                        data_loader=train_loader,
                        noise_multiplier=args.sigma,
                        max_grad_norm=args.max_per_sample_grad_norm,
                    )

                for epoch in range(1, args.epochs + 1):
                    custom, adam, mean_custom_grads_norms, regular_norms, mean_norm = train(
                        args, model, device, train_loader, optimizer, privacy_engine, epoch
                    )
                    test_accuracy = test(model, device, test_loader)
                    data = [{
                        'epoch': epoch,
                        'use_random_sketching': False,
                        'accuracy': test_accuracy,
                        'compression_rate': 1,
                        'use_sketching': False,
                        'dp_scale': dp_scale,
                        'use_preconditioning': False,
                        'debias': debias,
                        'use_numertrick': False,
                        'annihil': False,
                        'seed': seed
                    }]
                    df_epoch = pd.DataFrame(data)
                    df_epoch.to_csv(
                        csv_filename,
                        mode='a',
                        header=not file_exists,
                        index=False
                    )
                    file_exists = True


if __name__ == "__main__":
    main()
