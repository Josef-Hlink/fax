#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts every indexed replay (see index_slp.py) of the .zip archives into pools
of MDS shards, one pool per bucket (nofox, onefox, twofox) and archive. This is the slow step,
and it only has to happen once per archive: after it, the archive is no longer needed.
build_dataset.py then puts together the datasets the agents train on from the pools.
"""

import hashlib
import multiprocessing as mp
import os
import shutil
import struct
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
from fax.utils.schema import MDS_DTYPE_STR_BY_COLUMN, NP_TYPE_BY_COLUMN, RANK_INDEX


POOL_DIR = 'pool'  # inside the directory of the MDS datasets

# archives opened by this worker process
_zip_archs: Dict[Path, ZipFile] = {}


FAILED = 'FAILED'  # in place of a pool part, for a replay that could not be converted


def pick_order(replay: Dict[str, Any], seed: int) -> Tuple[bool, bytes]:
    """
    The seeded order in which a bucket's replays are converted and picked for datasets;
    replays that already belong to a split come first.
    """
    digest = hashlib.md5(f'{seed}-{replay["file_name"]}'.encode()).digest()
    return replay['split'] is None, digest


def slp_to_mds(
    db_path: Path,
    zips: Sequence[Path],
    pool_dir: Path,
    seed: int,
    buckets: Sequence[str] = BUCKETS,
    part_size: int = 4096,
    n_workers: int = 16,
) -> None:
    """Convert every indexed replay of the given archives into pools of MDS shards.

    A pool holds one bucket of one archive, in parts of `part_size` replays; the index records
    where every replay ended up. A part is done once the index knows its replays, so a run that
    was interrupted goes on with the first part that is not.

    Args:
        db_path: The index made by index_slp.py.
        zips: The .zip archives to convert.
        pool_dir: Directory where the pools will be created.
        seed: Seed for the order of the replays, see `pick_order`.
        buckets: The buckets to convert.
        part_size: Number of replays per part.
        n_workers: Number of worker processes.
    """
    db = DataBase(db_path)
    db.create_replays_table()
    for arch_file in sorted(zips, key=lambda p: p.name):
        for bucket in buckets:
            replays = [
                r
                for r in db.replays_in_bucket(BUCKETS.index(bucket))
                if r['archive'] == arch_file.name
            ]
            assert all(r['member'] for r in replays), 'Index is out of date; rerun index_slp'
            replays.sort(key=lambda r: pick_order(r, seed))
            for start in range(0, len(replays), part_size):
                part = replays[start : start + part_size]
                if all(r['pool_part'] for r in part):
                    continue
                name = f'{bucket}/{arch_file.stem}/{start // part_size:04d}'
                if (pool_dir / name).exists():
                    shutil.rmtree(pool_dir / name)  # left by a run that was interrupted
                jobs = [
                    (arch_file, r['member'], r['file_name'], r['p1rank'], r['p2rank']) for r in part
                ]
                written = process_replays(jobs, pool_dir / name, n_workers)
                rows = [(file_name, name, pos) for pos, file_name in enumerate(written)]
                failed = {r['file_name'] for r in part} - set(written)
                db.set_pooled(rows + [(file_name, FAILED, None) for file_name in failed])
    db.close()


def hash_to_int32(data: str) -> int:
    hash_bytes = hashlib.md5(data.encode()).digest()  # Get 16-byte hash
    int32_value = struct.unpack('i', hash_bytes[:4])[0]  # Convert first 4 bytes to int32
    return int32_value


def process_job(
    job: Tuple[Path, str, str, Optional[str], Optional[str]],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Get a replay out of its archive, process it and add what only the index knows."""
    arch_file, member, file_name, p1rank, p2rank = job
    if arch_file not in _zip_archs:
        _zip_archs[arch_file] = ZipFile(arch_file, 'r')
    tmp_path = scratch_dir() / f'fax-convert-{os.getpid()}.slp'
    try:
        extract_member(_zip_archs[arch_file], member, tmp_path)
        sample = process_replay(tmp_path, replay_uuid=hash_to_int32(file_name))
    except Exception as e:
        logger.debug(f'Error reading {member} from {arch_file.name}: {e}')
        return file_name, None
    finally:
        tmp_path.unlink(missing_ok=True)
    if sample is not None:
        n_frames = len(sample['frame'])
        for player, rank in (('p1', p1rank), ('p2', p2rank)):
            sample[f'{player}_rank'] = np.full(n_frames, RANK_INDEX.get(rank or '', 0), np.int32)
    return file_name, sample


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


def process_replays(jobs: List[Tuple[Any, ...]], mds_dir: Path, n_workers: int) -> List[str]:
    """Process replays into an MDS dataset.
    Args:
        jobs: What `process_job` needs of every replay.
        mds_dir: Directory where the MDS dataset will be created.
        n_workers: Number of worker processes.
    Returns:
        The file names of the replays that were written, in the order of the dataset.
    """
    logger.info(f'Processing {len(jobs)} replays into MDS dataset at {mds_dir}...')
    mds_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    with MDSWriter(
        out=mds_dir.as_posix(),
        columns=MDS_DTYPE_STR_BY_COLUMN,
        compression='zstd',
        size_limit=1 << 31,  # Write 2GB shards, data is repetitive so compression is 10-20x
        exist_ok=True,
    ) as out:
        with mp.Pool(n_workers) as pool:
            results = pool.imap_unordered(process_job, jobs)
            for file_name, sample in tqdm(results, total=len(jobs), desc=f'Processing {mds_dir}'):
                if sample is not None:
                    out.write(sample)
                    written.append(file_name)
                else:
                    logger.debug(f'Skipping invalid replay {file_name}')
    logger.info(f'Wrote {len(written)} replays to {mds_dir}')
    return written


if __name__ == '__main__':
    exposed_args = {'PATHS': 'zips sql mds', 'BASE': 'seed debug'}
    parser = create_parser(exposed_args)
    parser.add_argument(
        '--buckets', nargs='+', choices=BUCKETS, default=list(BUCKETS), help='Buckets to convert.'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=max(1, mp.cpu_count() - 4),  # leave some cores free
        help='Number of worker processes.',
    )
    args = parser.parse_args()
    cfg = parse_args(args, __file__)

    slp_to_mds(
        cfg.paths.sql,
        cfg.paths.zips,
        cfg.paths.mds.expanduser().resolve() / POOL_DIR,
        seed=cfg.base.seed,
        buckets=args.buckets,
        n_workers=args.n_workers,
    )
