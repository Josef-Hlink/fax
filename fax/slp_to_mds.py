#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a directory of .slp files into an MDS dataset.
"""

import hashlib
import multiprocessing as mp
import random
import struct
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import melee
import numpy as np
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from fax.constants import NP_MASK_VALUE
from fax.database import DataBase
from fax.gamestate_utils import FrameData, extract_and_append_gamestate_inplace
from fax.schema import MDS_DTYPE_STR_BY_COLUMN, NP_TYPE_BY_COLUMN
from fax.slp_reader import parse_replay
from fax.utils import debug_enabled


def main(slp_dir: Path, db_path: Path, mds_dir: Path, n: int, w: int) -> None:
    # initialize tables
    db = setup_database(db_path)
    # parse and index .slp files
    parse_and_store(slp_dir, db, n)
    # remove faulty replays from disk
    remove_faulty_replays(slp_dir, db.get_faulty_replays())
    # split files into fox dittos, one-fox, and no-fox datasets
    twofox, onefox, nofox = split_on_fox(slp_dir, db)
    # first write the dittos (they don't need to be split in train/val)
    twofox_files = [slp_dir / f for f in twofox]
    random.shuffle(twofox_files)
    process_replays(twofox_files, mds_dir / 'twofox', n_workers=w)
    # then write the one-fox and no-fox datasets (they get split in train/val)
    for datasetname, filenames in [('onefox', onefox), ('nofox', nofox)]:
        files = list(slp_dir / f for f in filenames)
        random.shuffle(files)
        splits = split_train_val(files, split=0.95)
        for split, data in splits.items():
            split_output_dir = mds_dir / datasetname / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            process_replays(data, split_output_dir, n_workers=w)
    return


def setup_database(db_path: Path) -> DataBase:
    db = DataBase(db_path)
    db.create_replays_table()
    db.create_errors_table()
    logger.info(f'Created database at {db_path}')
    return db


def parse_and_store(slp_dir: Path, db: DataBase, n: int) -> None:
    """Parse .slp files in the given directory and index them into an SQLite database.
    Args:
        slp_dir: Directory containing .slp files to index.
        db: DataBase instance to use for storing records.
        n: Number of files to process. Default is -1 (process all files).
    """

    logger.info(f'Indexing .slp files in {slp_dir}...')

    # create iterator over .slp files
    iterator = slp_dir.rglob('*.slp')
    if n > 0:
        total = min(n, sum(1 for _ in slp_dir.rglob('*.slp')))
        iterator = islice(iterator, n)
    elif n == -1:
        total = sum(1 for _ in slp_dir.rglob('*.slp'))
        iterator = slp_dir.rglob('*.slp')
    else:
        raise ValueError('n must be -1 (process all files) or a positive integer')

    # if not in debug mode, wrap iterator in tqdm for sleeker UI
    if not debug_enabled():
        iterator = tqdm(iterator, desc=f'Indexing .slp files in {slp_dir}', total=total)

    # actually parse and store each file
    for slp_path in iterator:
        try:
            db.insert_replay(parse_replay(slp_path, parse_ranks=True, parse_stocks=True))
        except Exception as e:
            db.insert_error(str(slp_path.name), str(e))

    logger.info(f'Finished indexing .slp files into {db.db_path}')
    logger.info(f'{db.n_replays} replays successfully indexed')
    logger.info(f'{db.n_errors} files failed to parse')

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
        # Double step on first frame to match next controller state to current gamestate
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


def process_replays(replay_paths: list[Path], mds_dir: Path, n_workers: int) -> None:
    """Process a list of replay files into an MDS dataset.
    Args:
        replay_paths: List of paths to .slp replay files.
        mds_dir: Directory where the MDS dataset will be created.
        n_workers: Number of MP worker processes to use for processing replays.
    """

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
    import sys
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    from fax.paths import DEFAULT_DB_PATH, DEFAULT_MDS_DIR, LOG_DIR
    from fax.utils import setup_logger

    parser = ArgumentParser(
        description='Index a directory of .slp files into an SQLite database.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'slp_dir',
        type=Path,
        help='Directory containing .slp files to index.',
    )
    parser.add_argument(
        '--db_path',
        type=Path,
        default=DEFAULT_DB_PATH,
        help='Path where the SQLite database will be created. '
        + 'Defaults to <project_root>/data/index.db.',
    )
    parser.add_argument(
        '--mds_dir',
        type=Path,
        default=DEFAULT_MDS_DIR,
        help='Directory where the MDS dataset will be created. '
        + 'Defaults to <project_root>/data/mds.',
    )
    parser.add_argument(
        '-n',
        type=int,
        default=-1,
        help='Number of files to process. Default is -1 (process all files).',
    )
    parser.add_argument(
        '-w',
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help='Number of MP workers processes to use for processing replays. '
        + 'Defaults to number of CPU cores minus one.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling replays before processing.',
    )
    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug mode.')
    args = parser.parse_args()

    # resolve paths
    slp_dir = args.slp_dir.expanduser().resolve()
    assert slp_dir.is_dir(), f'{slp_dir} is not a directory'
    assert any(slp_dir.rglob('*.slp')), f'No .slp files found in {slp_dir}'

    db_path = args.db_path.expanduser().resolve()
    if db_path.exists():
        logger.error(f'{db_path} already exists; please delete it first')
        sys.exit(1)

    mds_dir = args.mds_dir.expanduser().resolve()
    if mds_dir.exists() and any(mds_dir.iterdir()):
        logger.error(f'{mds_dir} already exists and has contents; please delete it first')
        sys.exit(1)
    mds_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    setup_logger(LOG_DIR / 'slp_to_mds.log', args.debug)

    main(slp_dir, db_path, mds_dir, args.n, args.w)
