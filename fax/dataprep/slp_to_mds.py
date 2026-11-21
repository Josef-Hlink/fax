#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts indexed replays (see index_slp.py) into 3 MDS datasets, one per bucket
(nofox, onefox, twofox), each with a train and a val split of the requested sizes.
The replays are read straight from the .zip archives.
The resulting MDS datasets can be used for training the agents.
"""

import hashlib
import multiprocessing as mp
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import melee
import numpy as np
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.dataprep.database import DataBase
from fax.dataprep.index_slp import BUCKETS, scratch_dir
from fax.dataprep.slp_reader import extract_member
from fax.utils.constants import NP_MASK_VALUE
from fax.utils.gamestate_utils import FrameData, extract_and_append_gamestate_inplace
from fax.utils.schema import MDS_DTYPE_STR_BY_COLUMN, NP_TYPE_BY_COLUMN


# archives opened by this worker process
_zip_archs: Dict[Path, ZipFile] = {}


def slp_to_mds(
    db_path: Path,
    zips: Sequence[Path],
    mds_dir: Path,
    n_train: int,
    n_val: int,
    seed: int,
    buckets: Sequence[str] = BUCKETS,
) -> None:
    """Convert indexed replays into MDS datasets, reading them straight from the archives.
    Args:
        db_path: The index made by index_slp.py.
        zips: The .zip archives the index refers to.
        mds_dir: Directory where the MDS datasets will be created.
        n_train: Number of replays per bucket for the train split.
        n_val: Number of replays per bucket for the val split.
        seed: Seed for picking the replays.
        buckets: The buckets to convert.
    """
    zip_by_name = {path.name: path for path in zips}
    db = DataBase(db_path)
    db.create_replays_table()
    for bucket in buckets:
        splits = pick_replays(db, BUCKETS.index(bucket), n_train, n_val, seed)
        for split, replays in splits.items():
            missing = {r['archive'] for r in replays} - set(zip_by_name)
            assert not missing, f'Archives {missing} are not among --zips'
            jobs = [
                (zip_by_name[r['archive']], r['member'], r['file_name'], r['p1rank'], r['p2rank'])
                for r in replays
            ]
            split_output_dir = mds_dir / bucket / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            process_replays(jobs, split_output_dir)
    db.close()


def pick_replays(
    db: DataBase, bucket: int, n_train: int, n_val: int, seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pick the replays of a bucket's train and val split, in a seeded order.

    The index remembers which split a replay was picked for, and those come first: a bigger
    dataset contains the smaller one, and a replay never moves from one split to the other.
    """
    replays = db.replays_in_bucket(bucket)
    assert all(r['member'] for r in replays), 'Index has replays without a place; rerun index_slp'
    replays.sort(key=lambda r: hashlib.md5(f'{seed}-{r["file_name"]}'.encode()).digest())
    free = [r for r in replays if r['split'] is None]
    splits = {}
    for split, n in (('train', n_train), ('val', n_val)):
        picked = [r for r in replays if r['split'] == split][:n]
        new, free = free[: n - len(picked)], free[n - len(picked) :]
        db.set_split([r['file_name'] for r in new], split)
        splits[split] = picked + new
        if len(splits[split]) < n:
            logger.warning(f'{BUCKETS[bucket]}/{split}: only {len(splits[split])} of {n} replays')
    return splits


def hash_to_int32(data: str) -> int:
    hash_bytes = hashlib.md5(data.encode()).digest()  # Get 16-byte hash
    int32_value = struct.unpack('i', hash_bytes[:4])[0]  # Convert first 4 bytes to int32
    return int32_value


def process_job(
    job: Tuple[Path, str, str, Optional[str], Optional[str]],
) -> Optional[Dict[str, Any]]:
    """Get a replay out of its archive and process it."""
    arch_file, member, file_name, p1rank, p2rank = job
    if arch_file not in _zip_archs:
        _zip_archs[arch_file] = ZipFile(arch_file, 'r')
    tmp_path = scratch_dir() / f'fax-convert-{os.getpid()}.slp'
    try:
        extract_member(_zip_archs[arch_file], member, tmp_path)
        sample = process_replay(tmp_path, replay_uuid=hash_to_int32(file_name))
    except Exception as e:
        logger.debug(f'Error reading {member} from {arch_file.name}: {e}')
        return None
    finally:
        tmp_path.unlink(missing_ok=True)
    return sample


def process_replay(replay_path: Path, replay_uuid: int) -> Optional[Dict[str, Any]]:
    """Process a single replay file and extract frame data.
    Args:
        replay_path: Path to the .slp replay file.
        replay_uuid: Number to tell the replay's frames from those of others.
    Returns:
        A dictionary containing frame data arrays, or None if processing failed.

    Largely adapted from https://github.com/ericyuegu/hal
    """
    frame_data: FrameData = defaultdict(list)
    try:
        # netplay replays contain every frame that was rolled back twice (or more): first as it was
        # predicted, later as it really happened; by default libmelee only returns the first one
        # (which is what a live bot needs), we want all of them so we can keep the final version
        console = melee.Console(
            path=str(replay_path),
            is_dolphin=False,
            allow_old_version=True,
            skip_rollback_frames=False,
        )
        console.connect()
    except Exception as e:
        logger.debug(f'Error connecting to console for {replay_path}: {e}')
        return None

    try:
        # read the whole game, letting later versions of a frame replace earlier ones
        gamestates: List[melee.GameState] = []
        first_frame: Optional[int] = None
        gamestate = console.step()
        while gamestate is not None:
            if first_frame is None:
                first_frame = gamestate.frame
            del gamestates[gamestate.frame - first_frame :]
            assert len(gamestates) == gamestate.frame - first_frame, 'gap in frame numbers'
            gamestates.append(gamestate)
            gamestate = console.step()

        # controller state lives in the gamestate that results from it, so we pair every
        # gamestate with the next one to get "what to press given the current frame"
        for curr_gamestate, next_gamestate in zip(gamestates, [*gamestates[1:], None]):
            frame_data = extract_and_append_gamestate_inplace(
                frame_data_by_field=frame_data,
                curr_gamestate=curr_gamestate,
                next_gamestate=next_gamestate,
                replay_uuid=replay_uuid,
            )
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


def process_replays(jobs: List[Tuple[Any, ...]], mds_dir: Path) -> None:
    """Process replays into an MDS dataset.
    Args:
        jobs: What `process_job` needs of every replay.
        mds_dir: Directory where the MDS dataset will be created.
    """
    replay_paths = jobs

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
            samples = pool.imap_unordered(process_job, replay_paths)
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
    exposed_args = {
        'PATHS': 'zips sql mds',
        'BASE': 'seed debug',
        'TRAINING': 'n-samples n-val-samples',
    }
    parser = create_parser(exposed_args)
    parser.add_argument(
        '--buckets', nargs='+', choices=BUCKETS, default=list(BUCKETS), help='Buckets to convert.'
    )
    args = parser.parse_args()
    cfg = parse_args(args, __file__)

    mds_dir = cfg.paths.mds.expanduser().resolve()
    for bucket in args.buckets:
        if (mds_dir / bucket).exists() and any((mds_dir / bucket).iterdir()):
            logger.error(f'{mds_dir / bucket} already has contents; please delete it first')
            sys.exit(1)
    mds_dir.mkdir(parents=True, exist_ok=True)

    slp_to_mds(
        cfg.paths.sql,
        cfg.paths.zips,
        mds_dir,
        n_train=cfg.training.n_samples,
        n_val=cfg.training.n_val_samples,
        seed=cfg.base.seed,
        buckets=args.buckets,
    )
