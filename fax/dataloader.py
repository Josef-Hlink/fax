from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence, Tuple

import torch
from streaming import StreamingDataLoader, StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from tensordict import TensorDict

from fax.config import create_parser, parse_args, CFG
from fax.constants import Player
from fax.dataprep.stats import load_dataset_stats
from fax.processing.preprocessor import Preprocessor, convert_ndarray_to_tensordict


def collate_tensordicts(batch: Sequence[TensorDict]) -> TensorDict:
    """Custom collate function for TensorDict because PyTorch type routing doesn't know about it yet."""
    return torch.stack(batch)  # type: ignore


class FAXStreamingDataset(StreamingDataset):
    """
    Streaming dataset that loads MDS data and applies preprocessing pipeline.

    Each sample from MDS is a full episode. This dataset:
    1. Loads episode data from MDS
    2. Converts to TensorDict
    3. Randomly samples trajectory_sampling_len subsequences
    4. Applies preprocessing (normalization, feature engineering)
    5. Applies temporal offsets
    6. Returns training-ready input/target pairs
    """

    def __init__(self, cfg: CFG, split: str = 'train'):
        """Fax streaming dataset constructor."""

        n_samples = cfg.training.n_samples if split == 'train' else cfg.training.n_eval_samples
        super().__init__(
            local=Path(cfg.paths.mds / 'onefox').expanduser().as_posix(),
            split=split,
            shuffle=True,
            batch_size=cfg.training.batch_size,
            epoch_size=n_samples,
        )

        self.cfg = cfg
        self.preprocessor = Preprocessor(
            cfg, load_dataset_stats(cfg.paths.mds / 'onefox')
        )  # TODO: parameterize dataset name
        self.is_train = split == 'train'  # for reproducible sampling during validation

    def __getitem__(self, idx: Any) -> TensorDict:
        # Get raw episode data from MDS
        episode_data = super().__getitem__(idx)

        # Convert numpy arrays to TensorDict
        episode_td = convert_ndarray_to_tensordict(episode_data)

        # Sample a subsequence for training
        sample_td = self.preprocessor.sample_from_episode(episode_td, debug=not self.is_train)

        # Randomly choose ego player (p1 or p2)
        ego: Player = random.choice(['p1', 'p2']) if self.is_train else 'p1'

        # Preprocess inputs and targets
        inputs_td = self.preprocessor.preprocess_inputs(sample_td, ego)
        targets_td = self.preprocessor.preprocess_targets(sample_td, ego)

        # Apply temporal offsets
        inputs_td = self.preprocessor.offset_inputs(inputs_td)
        targets_td = self.preprocessor.offset_targets(targets_td)

        # Combine into single TensorDict for training
        payload: dict[str, Any] = {
            'inputs': inputs_td,
            'targets': targets_td,
        }
        return TensorDict(payload, batch_size=())


def get_dataloaders(cfg: CFG) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        cfg: Training configuration containing data directory and batch size

    Returns:
        Tuple of (train_loader, val_loader)
    """

    # Clean stale shared memory
    clean_stale_shared_memory()

    # Create datasets
    train_dataset = FAXStreamingDataset(cfg=cfg, split='train')
    val_dataset = FAXStreamingDataset(cfg=cfg, split='val')

    # Create dataloaders
    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_tensordicts,
        num_workers=cfg.training.n_dataworkers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    val_loader = StreamingDataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_tensordicts,
        num_workers=cfg.training.n_dataworkers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    return train_loader, val_loader


if __name__ == '__main__':
    exposed_args = {'PATHS': 'mds', 'TRAINING': 'batch_size', 'MODEL': 'seq_len'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    train_loader, val_loader = get_dataloaders(cfg=cfg)

    for i, batch in enumerate(train_loader):
        print(f'Batch #{i + 1}:')
        print(f'  batch.shape: {batch.shape}')
        print(f'  batch.keys(): {list(batch.keys())}')

        inputs = batch['inputs']
        targets = batch['targets']

        print(f'  inputs.keys(): {list(inputs.keys())}')
        print(f'  targets.keys(): {list(targets.keys())}')

        # Print shapes of a few key features
        for key in ['stage', 'ego_character', 'gamestate', 'controller']:
            if key in inputs:
                print(f'  inputs[{key}].shape: {inputs[key].shape}')

        for key in ['main_stick', 'c_stick', 'buttons']:
            if key in targets:
                print(f'  targets[{key}].shape: {targets[key].shape}')

        if i >= 1:  # Test just 2 batches
            break

    print('Dataloader test completed successfully!')
