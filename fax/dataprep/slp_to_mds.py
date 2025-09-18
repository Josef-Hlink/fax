#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a directory of (bucketed) .slp files into 3 MDS datasets.
Each bucket (nofox, onefox, twofox) is split into train and validation sets
    based on how many training samples are specified.
The resulting MDS datasets can be used for training the agents.
"""

import hashlib
import multiprocessing as mp
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import melee
import numpy as np
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.utils.constants import NP_MASK_VALUE
from fax.utils.gamestate_utils import FrameData, extract_and_append_gamestate_inplace
from fax.utils.schema import MDS_DTYPE_STR_BY_COLUMN, NP_TYPE_BY_COLUMN


def slp_to_mds(slp_dir: Path, mds_dir: Path, split_idx: int) -> None:
    """Convert a directory of .slp files into an MDS dataset.
    Args:
        slp_dir: Directory containing .slp files to convert.
        mds_dir: Directory where the MDS datasets will be created.
    """
    for bucket in ['nofox', 'onefox', 'twofox']:
        files = list((slp_dir / bucket).glob('*.slp'))
        random.shuffle(files)
        splits = split_train_val(files, split_idx)
        for split, data in splits.items():
            split_output_dir = mds_dir / bucket / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            process_replays(data, split_output_dir)
    return


def hash_to_int32(data: str) -> int:
    hash_bytes = hashlib.md5(data.encode()).digest()  # Get 16-byte hash
    int32_value = struct.unpack('i', hash_bytes[:4])[0]  # Convert first 4 bytes to int32
    return int32_value


def split_train_val(input_paths: List[Path], split_idx: int) -> Dict[str, List[Path]]:
    """Split input paths into train and validation sets."""
    return {'train': input_paths[:split_idx], 'val': input_paths[split_idx:]}


def process_replay(replay_path: Path) -> Optional[Dict[str, Any]]:
    """Process a single replay file and extract frame data.
    Args:
        replay_path: Path to the .slp replay file.
    Returns:
        A dictionary containing frame data arrays, or None if processing failed.

    Largely adapted from https://github.com/ericyuegu/hal
    """
    frame_data: FrameData = defaultdict(list)
    try:
        console = melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.debug(f'Error connecting to console for {replay_path}: {e}')
        return None

    replay_uuid = hash_to_int32(str(replay_path))

    try:
        # double step on first frame to match next controller state to current gamestate
        curr_gamestate = console.step()
        while curr_gamestate is not None:
            next_gamestate = console.step()
            frame_data = extract_and_append_gamestate_inplace(
                frame_data_by_field=frame_data,
                curr_gamestate=curr_gamestate,
                next_gamestate=next_gamestate,
                replay_uuid=replay_uuid,
            )
            curr_gamestate = next_gamestate
            if curr_gamestate is None:
                break
    except AssertionError as e:
        logger.trace(f'Skipping replay {replay_path}: {e}')
        return None
    except Exception as e:
        logger.debug(f'Error processing replay {replay_path}: {e}')
        return None
    finally:
        console.stop()

    sample = {}
    for key, dtype in NP_TYPE_BY_COLUMN.items():
        if key in frame_data:
            array = [x if x is not None else NP_MASK_VALUE for x in frame_data[key]]
            sample[key] = np.array(array, dtype=dtype)

    sample['replay_uuid'] = np.array([replay_uuid] * len(frame_data['frame']), dtype=np.int32)
    return sample


def process_replays(replay_paths: list[Path], mds_dir: Path) -> None:
    """Process a list of replay files into an MDS dataset.
    Args:
        replay_paths: List of paths to .slp replay files.
        mds_dir: Directory where the MDS dataset will be created.
        n_workers: Number of MP worker processes to use for processing replays.
    """

    n_workers = max(1, mp.cpu_count() - 4)  # leave some cores free
    logger.info(f'Processing {len(replay_paths)} replays into MDS dataset at {mds_dir}...')

    processed = 0
    with MDSWriter(
        out=mds_dir.as_posix(),
        columns=MDS_DTYPE_STR_BY_COLUMN,
        compression='zstd',
        size_limit=1 << 31,  # Write 2GB shards, data is repetitive so compression is 10-20x
        exist_ok=True,
    ) as out:
        with mp.Pool(n_workers) as pool:
            samples = pool.imap_unordered(process_replay, replay_paths)
            for sample in tqdm(
                samples, total=len(replay_paths), desc=f'Processing {mds_dir.name} split'
            ):
                if sample is not None:
                    processed += 1
                    out.write(sample)
                else:
                    logger.debug('Skipping invalid replay sample')
    logger.info(f'Wrote {processed} replays to {mds_dir}')
    return


if __name__ == '__main__':
    exposed_args = {'PATHS': 'slp mds', 'TRAINING': 'n-samples', 'BASE': 'seed debug'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)

    # resolve paths
    slp_dir = cfg.paths.slp.expanduser().resolve()
    assert slp_dir.is_dir(), f'{slp_dir} is not a directory'

    mds_dir = cfg.paths.mds.expanduser().resolve()
    if mds_dir.exists() and any(mds_dir.iterdir()):
        logger.error(f'{mds_dir} already exists and has contents; please delete it first')
        sys.exit(1)
    mds_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.base.seed)

    slp_to_mds(slp_dir, mds_dir, split_idx=cfg.training.n_samples)
