#!/usr/bin/env python3
"""
Training script for FAX - Fox imitation learning on Melee replays.

This script brings together the dataloader, model, and training loop
to demonstrate how all components interact.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fax.config import DataConfig, TrainConfig
from fax.dataloader import get_dataloaders
from fax.model import Model, GPTConfig
from fax.processing.preprocessor import Preprocessor


TRAIN_LOSSES = []
VAL_LOSSES = []


def compute_loss(outputs, targets):
    """
    Compute loss for each controller head.

    Args:
        outputs: Model predictions (TensorDict with keys: main_stick, c_stick, buttons, shoulder)
        targets: Ground truth targets (TensorDict with same keys)

    Returns:
        Total loss and individual losses dict
    """
    losses = {}

    # Cross-entropy loss for each head (all are categorical)
    for head_name in ['main_stick', 'c_stick', 'buttons', 'shoulder']:
        if head_name in outputs and head_name in targets:
            # outputs[head_name]: (B, L, num_classes)
            # targets[head_name]: (B, L, num_classes) - one-hot encoded

            # Convert one-hot targets to class indices
            target_indices = targets[head_name].argmax(dim=-1)  # (B, L)

            # Reshape for cross-entropy: (B*L, num_classes) and (B*L,)
            B, L, C = outputs[head_name].shape
            pred_flat = outputs[head_name].view(B * L, C)
            target_flat = target_indices.view(B * L)

            losses[head_name] = F.cross_entropy(pred_flat, target_flat)

    # Total loss is sum of individual losses
    total_loss = sum(losses.values())

    return total_loss, losses


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_losses = {}
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss, losses = compute_loss(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate losses
        total_loss += loss.item()
        for head_name, head_loss in losses.items():
            if head_name not in total_losses:
                total_losses[head_name] = 0.0
            total_losses[head_name] += head_loss.item()

        num_batches += 1

        # Print progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch}, Batch {batch_idx:3d}: '
                f'Loss {loss.item():.4f}, '
                f'LR {scheduler.get_last_lr()[0]:.6f}, '
                f'Time {elapsed:.1f}s'
            )

        # store loss in global list
        TRAIN_LOSSES.append(loss.item())

    # Average losses
    avg_loss = total_loss / num_batches
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_loss, avg_losses


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_losses = {}
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(inputs)
            loss, losses = compute_loss(outputs, targets)

            total_loss += loss.item()
            for head_name, head_loss in losses.items():
                if head_name not in total_losses:
                    total_losses[head_name] = 0.0
                total_losses[head_name] += head_loss.item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    VAL_LOSSES.append(avg_loss)

    return avg_loss, avg_losses


def main():
    parser = argparse.ArgumentParser(description='Train FAX model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to MDS data directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--model-dim', type=int, default=256, help='Model embedding dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Using device: {device}')

    # Create configs
    data_config = DataConfig(dir=args.data_dir, seq_len=args.seq_len)
    train_config = TrainConfig(
        data=data_config, batch_size=args.batch_size, lr=args.lr, n_epochs=args.epochs
    )

    # Create model
    preprocessor = Preprocessor(data_config)
    gpt_config = GPTConfig(
        block_size=args.seq_len,
        n_embd=args.model_dim,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        dropout=0.1,
        bias=True,
    )
    model = Model(preprocessor=preprocessor, gpt_config=gpt_config)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {total_params:,} total parameters, {trainable_params:,} trainable')

    # Create dataloaders
    print('Creating dataloaders...')
    train_loader, val_loader = get_dataloaders(train_config)

    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f'Training for {args.epochs} epochs, {len(train_loader)} batches per epoch')
    print(f'Total training steps: {total_steps}')

    # Training loop
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')

        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch + 1
        )

        # Validate
        val_loss, val_losses = validate(model, val_loader, device)

        # Print epoch summary
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss:   {val_loss:.4f}')
        print(f'  Train Losses by Head: {train_losses}')
        print(f'  Val Losses by Head:   {val_losses}')

    print('\nTraining completed!')
    # save losses to npy files
    np.save('../data/train_losses.npy', np.array(TRAIN_LOSSES))
    np.save('../data/val_losses.npy', np.array(VAL_LOSSES))

    # Save the final model
    torch.save(model.state_dict(), '../data/fax_model_final.pth')


if __name__ == '__main__':
    main()
