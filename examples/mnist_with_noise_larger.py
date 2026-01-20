#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus_local import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

from opacus_local.optimizers.DOME_optim import AdamCorr

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
torch.set_default_dtype(torch.float64)


# --------- ResNet-8 with GroupNorm for MNIST --------- #
class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(num_groups, planes),
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
    """

    def __init__(self, num_classes=10, num_groups=8):
        super().__init__()
        self.in_planes = 16

        # MNIST: single-channel input
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, 16)

        # Stages (like CIFAR-style ResNet, but adapted to 28x28)
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
        # x: [B, 1, 28, 28]
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)

        out = self.layer1(out)  # 28x28
        out = self.layer2(out)  # 14x14
        out = self.layer3(out)  # 7x7

        out = self.avgpool(out)  # [B, 64, 1, 1]
        out = torch.flatten(out, 1)  # [B, 64]
        out = self.fc(out)  # [B, 10]
        return out

    def name(self):
        return "ResNet8GN"


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
        T_0=47,  # e.g. restart every 100 batches
        T_mult=1,  # keep the same period after each restart
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=469)

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
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


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
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
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
        default=40,
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
        help="Disable privacy training and just train with vanilla SGD",
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
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    csv_filename = "training_metrics_mnist_noise_2411.csv"
    file_exists = os.path.isfile(csv_filename)

    debias = True
    debias = False
    for seed in [2, 3, 4]:
        # for seed in [2]:
        set_seed(seed)
        # for dp_scale in [5.0]:
        # for compression_rate in [1000, 25, 10, 500, 250, 100, 50]:
        # for dp_scale in [0,5,10,20,30]:
        # for dp_scale in [0,2,3.5,5,7.5,10]:
        for dp_scale in [0]:
            # for dp_scale in [0.0]:
            # for dp_scale in [0.0]:
            # for dp_scale in [3.0, 4.0]:
            # for dp_scale in [0.0]:
            # for dp_scale in [0.1]:
            for compression_rate in [50,100,200,500,1000]:
                # for compression_rate in [250]:
                # for compression_rate in [100]:
                for use_random_sketch in [True, False]:
                    # for use_random_sketch in [True]:
                    annihil_list = []
                    if not use_random_sketch:
                        annihil_list = [True]
                    else:
                        annihil_list = [True]
                    for annihil in annihil_list:
                        # for use_random_sketch in [False]:
                        # for use_random_sketch in [False]:
                        if not use_random_sketch:
                            use_preconditioning_list = [False]
                            # use_preconditioning_list = [True]
                        else:
                            use_preconditioning_list = [False]
                        for use_preconditioning in use_preconditioning_list:
                            for _ in range(args.n_runs):
                                try:
                                    model = ResNet8GN().to(device)
                                    use_sketching = True

                                    nb_params = sum(
                                        p.numel() for p in model.parameters() if p.requires_grad
                                    )

                                    rank = nb_params // compression_rate

                                    optimizer = AdamCorr(
                                        model.parameters(),
                                        lr=0.001,
                                        dp_batch_size=64,
                                        dp_noise_multiplier=1,
                                        dp_l2_norm_clip=1,
                                        eps_root=1e-10,
                                        numel=nb_params,
                                        rank=rank,
                                        random_sketch=use_random_sketch,
                                        use_sketching=use_sketching,
                                        use_preconditioning=use_preconditioning,
                                        debias=debias,
                                        dp_scale=dp_scale,
                                        use_numertrick=False,
                                        annihil=annihil,
                                    )
                                    privacy_engine = None

                                    if not args.disable_dp:
                                        privacy_engine = PrivacyEngine(
                                            secure_mode=args.secure_rng
                                        )
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
                                            train_loader,
                                            optimizer,
                                            privacy_engine,
                                            epoch,
                                        )
                                        test_accuracy = test(
                                            model, device, test_loader
                                        )
                                        data = [
                                            {
                                                "epoch": epoch,
                                                "use_random_sketching": use_random_sketch,
                                                "accuracy": test_accuracy,
                                                "compression_rate": compression_rate,
                                                "use_sketching": use_sketching,
                                                "dp_scale": dp_scale,
                                                "use_preconditioning": use_preconditioning,
                                                "debias": debias,
                                                "use_numertrick": False,
                                                "annihil": annihil,
                                                "seed": seed,
                                            }
                                        ]
                                        df_epoch = pd.DataFrame(data)
                                        df_epoch.to_csv(
                                            csv_filename,
                                            mode="a",
                                            header=not file_exists,
                                            index=False,
                                        )
                                        # after first write, make sure we don’t rewrite the header
                                        file_exists = True
                                except Exception as error:
                                    print("An exception occurred ", error)

    debias = False
    for seed in [2, 3, 4]:
        for dp_scale in [1]:
            for _ in range(args.n_runs):
                compression_rate = 100
                use_random_sketch = False
                use_sketching = False
                use_preconditioning = False
                model = ResNet8GN().to(device)
                nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                rank = 26000 // 1000

                optimizer = AdamCorr(
                    model.parameters(),
                    dp_batch_size=128,
                    dp_noise_multiplier=1,
                    dp_l2_norm_clip=1,
                    eps_root=1e-8,
                    numel=nb_params,
                    rank=rank,
                    random_sketch=use_random_sketch,
                    use_sketching=use_sketching,
                    use_preconditioning=use_preconditioning,
                    debias=debias,
                    dp_scale=dp_scale,
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
                        train_loader,
                        optimizer,
                        privacy_engine,
                        epoch,
                    )
                    test_accuracy = test(model, device, test_loader)
                    data = [
                        {
                            "epoch": epoch,
                            "use_random_sketching": False,
                            "accuracy": test_accuracy,
                            "compression_rate": 1,
                            "use_sketching": False,
                            "dp_scale": dp_scale,
                            "use_preconditioning": False,
                            "debias": debias,
                            "use_numertrick": False,
                            "annihil": False,
                            "seed": seed,
                        }
                    ]
                    df_epoch = pd.DataFrame(data)
                    df_epoch.to_csv(
                        csv_filename,
                        mode="a",
                        header=not file_exists,
                        index=False,
                    )
                    # after first write, make sure we don’t rewrite the header
                    file_exists = True


if __name__ == "__main__":
    main()
