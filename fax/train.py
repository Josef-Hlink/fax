#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for FAX - Fox imitation learning on Melee replays.

This script brings together the dataloader, model, and training loop
to demonstrate how all components interact.
"""

import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fax.config import Config, create_parser, parse_args
from fax.dataloader import get_dataloaders
from fax.model import Model
from fax.paths import LOG_DIR
from fax.processing.preprocessor import Preprocessor
from fax.utils import setup_logger


def main(config: Config):
    """Main training loop."""

    random.seed(config.seed)
    device = torch.device('cuda' if config.n_gpus > 0 else 'cpu')

    preprocessor = Preprocessor(config)
    model = Model(preprocessor=preprocessor, config=config)
    model = model.to(device)
    logger.debug(f'Model: {sum(p.numel() for p in model.parameters()):,} total parameters')

    train_loader, val_loader = get_dataloaders(config)

    # create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.wd, betas=(config.b1, config.b2)
    )
    total_steps = len(train_loader) * config.n_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    logger.info(f'Training for {config.n_epochs} epochs, {len(train_loader)} batches per epoch')
    logger.info(f'Total training steps: {total_steps}')

    # training loop
    for epoch in range(config.n_epochs):
        tl = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        vl = validate(model, val_loader, device)
        logger.info(f'epoch {epoch + 1}: avg. {tl=:.4f}, {vl:.4f}')

    logger.info('\nTraining completed!')


def compute_loss(outputs, targets) -> torch.Tensor:
    """
    Compute loss for each controller head.

    Args:
        outputs: Model predictions (TensorDict with keys: main_stick, c_stick, buttons, shoulder)
        targets: Ground truth targets (TensorDict with same keys)

    Returns:
        Total loss as a single scalar tensor.
    """
    losses = {}

    # cross-entropy loss for each head (all are categorical)
    for head_name in ['main_stick', 'c_stick', 'buttons', 'shoulder']:
        # convert one-hot targets to class indices
        target_indices = targets[head_name].argmax(dim=-1)  # (B, L)

        # reshape for cross-entropy: (B*L, num_classes) and (B*L,)
        B, L, C = outputs[head_name].shape
        pred_flat = outputs[head_name].view(B * L, C)
        target_flat = target_indices.view(B * L)

        losses[head_name] = F.cross_entropy(pred_flat, target_flat)

    # total loss is sum of individual losses
    return torch.stack(list(losses.values())).sum()


def train_epoch(
    model: Model,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for sample in (pbar := tqdm(train_loader, desc=f'Training Epoch {epoch}')):
        # move to device
        inputs = sample['inputs'].to(device)
        targets = sample['targets'].to(device)
        # forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        # compute loss
        loss = compute_loss(outputs, targets)
        # backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        # accumulate and log loss
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    # average loss
    return total_loss / len(train_loader)


def validate(model: Model, val_loader: DataLoader, device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = create_parser(Config, parser)
    config = parse_args(Config, parser.parse_args())
    setup_logger(LOG_DIR / 'training.log', config.debug)
    main(config)
