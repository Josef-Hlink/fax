from __future__ import annotations

import random
from typing import Any, Optional, Sequence, Tuple

import torch
from streaming import StreamingDataLoader, StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from tensordict import TensorDict

from fax.config import CFG, create_parser, parse_args
from fax.constants import MATCHUP_TO_BUCKET, Player
from fax.processing.preprocessor import Preprocessor, convert_ndarray_to_tensordict

FOX = 2  # character ID for Fox


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

    def __init__(self, cfg: CFG, matchup: str, split: Optional[str] = None):
        """Fax streaming dataset constructor."""

        if split == 'train':
            n_samples = cfg.training.n_samples
        elif split is None or split == 'val':
            n_samples = cfg.training.n_val_samples
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or None.")

        self.matchup = matchup  # this will be used in __getitem__
        bucket = cfg.paths.mds / MATCHUP_TO_BUCKET[matchup]
        super().__init__(
            local=bucket.expanduser().as_posix(),
            split=split,
            shuffle=True,
            batch_size=cfg.training.batch_size,
            epoch_size=n_samples,
        )

        self.cfg = cfg
        self.preprocessor = Preprocessor(cfg)

    def __getitem__(self, idx: Any) -> TensorDict:
        # get raw episode data from MDS
        episode_data = super().__getitem__(idx)

        # convert numpy arrays to TensorDict
        episode_td = convert_ndarray_to_tensordict(episode_data)

        # sample a subsequence for training
        sample_td = self.preprocessor.sample_from_episode(episode_td)

        # determine ego player
        if self.matchup in ['FvF', 'XvX']:
            ego: Player = random.choice(['p1', 'p2'])
        elif self.matchup == 'FvX':
            ego: Player = 'p1' if sample_td['p1_character'][0].item() == FOX else 'p2'
        else:  # matchup == 'XvF'
            ego: Player = 'p1' if sample_td['p2_character'][0].item() == FOX else 'p2'

        # preprocess inputs and targets
        inputs_td = self.preprocessor.preprocess_inputs(sample_td, ego)
        targets_td = self.preprocessor.preprocess_targets(sample_td, ego)

        # apply temporal offsets
        inputs_td = self.preprocessor.offset_inputs(inputs_td)
        targets_td = self.preprocessor.offset_targets(targets_td)

        # combine into single TensorDict for training
        payload: dict[str, Any] = {
            'inputs': inputs_td,
            'targets': targets_td,
        }
        return TensorDict(payload, batch_size=())


def get_dataloaders(cfg: CFG, matchup: str) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        cfg: Training configuration containing batch size, num workers, etc.
        matchup: Matchup string to determine which MDS bucket to use.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # clean stale shared memory
    clean_stale_shared_memory()

    # create datasets
    train_dataset = FAXStreamingDataset(cfg, matchup, split='train')
    val_dataset = FAXStreamingDataset(cfg, matchup, split='val')

    # create dataloaders
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
    exposed_args = {'PATHS': 'mds', 'TRAINING': 'batch-size', 'MODEL': 'seq-len', 'EXP': 'matchup'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    train_loader, val_loader = get_dataloaders(cfg=cfg, matchup=cfg.exp.matchup)

    for i, batch in enumerate(train_loader):
        print(f'Batch #{i + 1}:')
        print(f'  batch.shape: {batch.shape}')
        print(f'  batch.keys(): {list(batch.keys())}')

        inputs = batch['inputs']
        targets = batch['targets']

        print(f'  inputs.keys(): {list(inputs.keys())}')
        print(f'  targets.keys(): {list(targets.keys())}')

        # print shapes of a few key features
        for key in ['stage', 'ego_character', 'gamestate', 'controller']:
            if key in inputs:
                print(f'  inputs[{key}].shape: {inputs[key].shape}')

        for key in ['main_stick', 'c_stick', 'buttons']:
            if key in targets:
                print(f'  targets[{key}].shape: {targets[key].shape}')

        if i >= 1:  # test just 2 batches
            break

    print('Dataloader test completed successfully!')
