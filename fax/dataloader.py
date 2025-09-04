# fax/dataloader.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence, Tuple

import torch
from streaming import StreamingDataset, StreamingDataLoader
from streaming.base.util import clean_stale_shared_memory
from tensordict import TensorDict

from fax.config import TrainConfig, DataConfig
from fax.constants import Player
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

    def __init__(
        self,
        local: str,
        remote: str | None = None,
        split: str = 'train',
        shuffle: bool = True,
        data_config: DataConfig | None = None,
        **kwargs,
    ):
        super().__init__(local=local, remote=remote, split=split, shuffle=shuffle, **kwargs)

        self.data_config = data_config or DataConfig()
        self.preprocessor = Preprocessor(self.data_config)

        # For reproducible sampling during validation
        self.is_train = split == 'train'

    def __getitem__(self, idx: int) -> TensorDict:
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
        return TensorDict(
            {
                'inputs': inputs_td,
                'targets': targets_td,
            },
            batch_size=(),
        )


def get_dataloaders(config: TrainConfig) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: Training configuration containing data directory and batch size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    batch_size = config.batch_size
    data_dir = Path(config.data.dir).expanduser()

    # Clean stale shared memory
    clean_stale_shared_memory()

    # Create datasets
    train_dataset = FAXStreamingDataset(
        local=str(data_dir),
        remote=None,
        split='train',
        shuffle=True,
        data_config=config.data,
        batch_size=batch_size,
    )

    val_dataset = FAXStreamingDataset(
        local=str(data_dir),
        remote=None,
        split='val',
        shuffle=False,
        data_config=config.data,
        batch_size=batch_size,
    )

    # Create dataloaders
    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_tensordicts,
        num_workers=2,  # TODO: make configurable
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    val_loader = StreamingDataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_tensordicts,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    return train_loader, val_loader


def save_dataloader_state(loader: StreamingDataLoader, path: Path) -> None:
    """Checkpoint the dataloader state to disk."""
    state = loader.state_dict()
    with path.open('wb') as f:
        torch.save(state, f)


def load_dataloader_state(loader: StreamingDataLoader, path: Path) -> None:
    """Load checkpointed dataloader state from disk."""
    state = torch.load(path)
    loader.load_state_dict(state)


if __name__ == '__main__':
    # Test the dataloader with preprocessing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to MDS data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    args = parser.parse_args()

    print('[main] Testing FAXStreamingDataset with preprocessing...')

    # Create config
    data_config = DataConfig(dir=args.data_dir, seq_len=args.seq_len)
    train_config = TrainConfig(data=data_config, batch_size=args.batch_size)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(train_config)

    print(f'[main] Created dataloaders with batch_size={args.batch_size}, seq_len={args.seq_len}')
    print(f'[main] Testing train loader...')

    for i, batch in enumerate(train_loader):
        print(f'[main] Batch #{i + 1}:')
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

    print('[main] Dataloader test completed successfully!')
