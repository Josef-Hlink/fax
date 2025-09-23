#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for FAX - Fox imitation learning on Melee replays.

This script brings together the dataloader, model, and training loop
to demonstrate how all components interact.
"""

import random

import torch
import torch.nn.functional as F
from loguru import logger
from randomname import get_name
from streaming import StreamingDataLoader
from tensordict import TensorDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fax.config import CFG, create_parser, parse_args
from fax.model import Model
from fax.processing.preprocessor import Preprocessor
from fax.training.dataloader import get_dataloaders
from fax.utils.writer import DummyWriter, WandbWriter


def train(cfg: CFG) -> None:
    logger.info('Starting training...')

    preprocessor = Preprocessor(cfg)
    model = Model(preprocessor, cfg)
    logger.info(f'Model has {sum(p.numel() for p in model.parameters()):,} parameters.')

    # get dataloaders
    own_train_loader, own_val_loader = get_dataloaders(cfg, cfg.training.matchup)
    fvf_train_loader, fvf_val_loader = get_dataloaders(cfg, 'FvF')

    # initialize trainer
    trainer = Trainer(cfg, model)

    best_val_loss = float('inf')
    for epoch in range(1, cfg.training.n_epochs + 1):
        train_loss = trainer.train_epoch(own_train_loader, epoch, cfg.training.n_epochs)
        own_val_loss = trainer.validate(own_val_loader)
        fvf_val_loss = trainer.validate(fvf_val_loader)

        logger.info(
            f'Epoch {epoch}: tl = {train_loss:.4f}, '
            + f'own vl = {own_val_loss:.4f}, '
            + f'FvF vl = {fvf_val_loss:.4f}'
        )
        trainer.writer.log({'val/own_loss': own_val_loss}, None, commit=False)
        trainer.writer.log({'val/fvf_loss': fvf_val_loss}, None, commit=False)

        # save best model
        if own_val_loss < best_val_loss:
            best_val_loss = own_val_loss
            torch.save(model.state_dict(), cfg.paths.runs / trainer.run_name / 'best_model.pth')
            logger.info(f'Saved new best model with Val Loss = {best_val_loss:.4f}')

    if cfg.training.n_finetune_epochs == 0:
        logger.info('No finetuning epochs specified, skipping finetuning.')
        trainer.writer.finish()
        logger.info('Training complete.')
        return

    # finetuning phase
    logger.info('Initial training regiment completed. Starting finetuning phase...')
    trainer.optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr * cfg.training.finetune_lr_frac,
        weight_decay=cfg.optim.wd,
        betas=(cfg.optim.b1, cfg.optim.b2),
    )
    total_steps = cfg.training.n_finetune_epochs * cfg.training.n_samples // cfg.training.batch_size
    trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=total_steps, eta_min=1e-6)
    best_val_loss = float('inf')
    for epoch in range(1, cfg.training.n_finetune_epochs + 1):
        train_loss = trainer.train_epoch(fvf_train_loader, epoch, cfg.training.n_finetune_epochs)
        fvf_val_loss = trainer.validate(fvf_val_loader)

        logger.info(f'Finetune epoch {epoch}: tl = {train_loss:.4f}, fvf vl = {fvf_val_loss:.4f}')
        trainer.writer.log({'val/fvf_loss': fvf_val_loss}, None, commit=False)

        # save best model
        if fvf_val_loss < best_val_loss:
            best_val_loss = fvf_val_loss
            torch.save(
                model.state_dict(), cfg.paths.runs / trainer.run_name / 'best_model_finetuned.pth'
            )
            logger.info(f'Saved new best finetuned model with Val Loss = {best_val_loss:.4f}')

    return


class Trainer(torch.nn.Module):
    def __init__(self, cfg: CFG, model: Model):
        super().__init__()
        self.cfg = cfg
        self.model = model

        self.run_name = f'{get_name()}-{random.randint(0, 1000):03d}'
        if not (cfg.paths.runs / self.run_name).exists():
            (cfg.paths.runs / self.run_name).mkdir(parents=True, exist_ok=True)
        logger.info(f'Run name: {self.run_name}')
        self.writer: WandbWriter | DummyWriter = DummyWriter(cfg, self.run_name)
        if cfg.base.wandb:
            self.writer = WandbWriter(cfg, self.run_name)
        random.seed(cfg.base.seed)

        self.device = torch.device('cuda' if cfg.base.n_gpus > 0 else 'cpu')
        self.model.to(self.device)
        self.to(self.device)

        lr, wd, b1, b2 = map(lambda x: getattr(cfg.optim, x), ['lr', 'wd', 'b1', 'b2'])
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))
        total_steps = cfg.training.n_epochs * cfg.training.n_samples // cfg.training.batch_size
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)

    def train_epoch(self, train_loader: StreamingDataLoader, epoch: int, n_epochs: int) -> float:
        """Train the model on the provided dataloader for one epoch.
        Args:
            train_loader: DataLoader providing training batches.
            epoch: Current epoch number (1-indexed).
            n_epochs: Total number of epochs.
        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        # clean_stale_shared_memory()

        for batch in tqdm(train_loader, desc=f'epoch {epoch}/{n_epochs}'):
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            self.writer.log({'train/loss': loss.item()}, None, commit=True)
            self.writer.log({'train/lr': self.scheduler.get_last_lr()[0]}, None, commit=False)

        return total_loss / len(train_loader)

    def validate(self, val_loader: StreamingDataLoader) -> float:
        """Evaluate the model on the provided validation dataloader.
        Args:
            val_loader: DataLoader providing validation batches.
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        # clean_stale_shared_memory()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'validating'):
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                outputs = self.model(inputs)
                loss = compute_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    @property
    def log_dir(self):
        return self.cfg.paths.logs


def compute_loss(outputs: TensorDict, targets: TensorDict) -> torch.Tensor:
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


if __name__ == '__main__':
    exposed_args = {
        'PATHS': 'mds',
        'BASE': 'seed debug wandb n-gpus',
        'MODEL': 'n-layers n-heads seq-len emb-dim dropout gamma',
        'OPTIM': 'lr wd b1 b2',
        'TRAINING': '*',  # NOTE: this exposes all exp args
    }
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    assert cfg.training.matchup is not None, 'Matchup must be specified.'
    train(cfg)
