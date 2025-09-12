#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a directory of .slp files into 3 MDS datasets:
  - fox dittos
  - one-fox games (split into train/val)
  - no-fox games (split into train/val)
"""

import sys
import hashlib
import multiprocessing as mp
import random
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import melee
import numpy as np
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.constants import NP_MASK_VALUE
from fax.dataprep.database import DataBase
from fax.gamestate_utils import FrameData, extract_and_append_gamestate_inplace
from fax.schema import MDS_DTYPE_STR_BY_COLUMN, NP_TYPE_BY_COLUMN


def slp_to_mds(slp_dir: Path, db_path: Path, mds_dir: Path, seq_len: int) -> None:
    """Convert a directory of .slp files into an MDS dataset.
    Args:
        slp_dir: Directory containing .slp files to convert.
        db_path: Path to the SQLite database file where the .slp files are indexed.
        mds_dir: Directory where the MDS datasets will be created.
        seq_len: Minimum sequence length (in frames) for a replay to be included.
    """

    db = DataBase(db_path)
    # remove faulty replays from disk
    faulty_replays = (
        db.get_corrupted_replays()
        + db.get_unfinished_replays()
        + db.get_short_replays(min_frames=seq_len)
    )
    remove_faulty_replays(slp_dir, faulty_replays)
    # split files into fox dittos, one-fox, and no-fox datasets
    twofox, onefox, nofox = split_on_fox(slp_dir, db)
    # first write the dittos (they don't need to be split in train/val)
    twofox_files = [slp_dir / f for f in twofox]
    random.shuffle(twofox_files)
    process_replays(twofox_files, mds_dir / 'twofox')
    # then write the one-fox and no-fox datasets (they get split in train/val)
    for datasetname, filenames in [('onefox', onefox), ('nofox', nofox)]:
        files = list(slp_dir / f for f in filenames)
        random.shuffle(files)
        splits = split_train_val(files, split=0.95)
        for split, data in splits.items():
            split_output_dir = mds_dir / datasetname / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            process_replays(data, split_output_dir)
    return


def remove_faulty_replays(slp_dir: Path, replays: List[str]) -> None:
    """Remove replay files that resulted in parse errors from the directory."""
    if not replays:
        logger.info('No faulty replays to remove')
        return
    logger.warning(f'Removing {len(replays)} faulty replay files from {slp_dir}...')
    for replay in replays:
        slp_path = slp_dir / replay
        if slp_path.exists():
            slp_path.unlink()
            logger.debug(f'Removed {slp_path}')
        else:
            logger.warning(f'File {slp_path} does not exist; cannot remove')
    logger.info('Finished removing faulty replay files')
    return


def split_on_fox(slp_dir: Path, db: DataBase) -> Tuple[set[str], set[str], set[str]]:
    """Split .slp files into fox dittos, one-fox, and no-fox datasets."""
    # all .slp files that still exist on disk are valid (problematic ones were deleted)
    all_valid_files = {f.name for f in slp_dir.rglob('*.slp')}
    logger.debug(f'Found {len(all_valid_files)} valid .slp files in {slp_dir}')
    # check index for all fox files and fox dittos
    indexed_fox = set(db.query_character('fox'))
    indexed_dittos = set(db.query_matchup('fox', 'fox'))
    logger.debug(f'Found {len(indexed_fox)} indexed fox files in database')
    logger.debug(f'Found {len(indexed_dittos)} fox ditto files in database')
    # set operations
    onefox_files = (indexed_fox - indexed_dittos) & all_valid_files
    logger.info(f'Arrived at {len(onefox_files)} valid one-fox replays')
    twofox_files = indexed_dittos & all_valid_files
    logger.info(f'Arrived at {len(twofox_files)} valid fox ditto replays')
    nofox_files = all_valid_files - indexed_fox
    logger.info(f'Arrived at {len(nofox_files)} valid no-fox replays')
    return twofox_files, onefox_files, nofox_files


def hash_to_int32(data: str) -> int:
    hash_bytes = hashlib.md5(data.encode()).digest()  # Get 16-byte hash
    int32_value = struct.unpack('i', hash_bytes[:4])[0]  # Convert first 4 bytes to int32
    return int32_value


def split_train_val(input_paths: List[Path], split: float = 0.95) -> Dict[str, List[Path]]:
    """Split input paths into train and validation sets."""
    split_idx = int(len(input_paths) * split)
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
    exposed_args = {'PATHS': 'slp sql mds', 'BASE': 'seed', 'MODEL': 'seq-len'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)

    # resolve paths
    slp_dir = cfg.paths.slp.expanduser().resolve()
    assert slp_dir.is_dir(), f'{slp_dir} is not a directory'
    assert any(slp_dir.rglob('*.slp')), f'No .slp files found in {slp_dir}'

    db_path = cfg.paths.sql.expanduser().resolve()
    if db_path.exists():
        logger.error(f'{db_path} already exists; please delete it first')
        sys.exit(1)

    mds_dir = cfg.paths.mds.expanduser().resolve()
    if mds_dir.exists() and any(mds_dir.iterdir()):
        logger.error(f'{mds_dir} already exists and has contents; please delete it first')
        sys.exit(1)
    mds_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.base.seed)

    slp_to_mds(slp_dir, db_path, mds_dir, cfg.model.seq_len)
