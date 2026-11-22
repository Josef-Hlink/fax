#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script puts together the MDS datasets the agents train on: per bucket (nofox, onefox,
twofox) a train and a val split of the requested sizes, copied out of the pools that
slp_to_mds.py made. No replay is parsed here, so making another dataset is cheap.
"""

import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from loguru import logger
from streaming import MDSWriter, StreamingDataset
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.dataprep.database import DataBase
from fax.dataprep.index_slp import BUCKETS
from fax.dataprep.slp_to_mds import FAILED, POOL_DIR, pick_order
from fax.utils.schema import MDS_DTYPE_STR_BY_COLUMN, UNUSED_COLUMNS


def pick_replays(
    db: DataBase, bucket: int, n_train: int, n_val: int, seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pick the replays of a bucket's train and val split from the pools, in the seeded order.

    The index remembers which split a replay was picked for, and those come first: a bigger
    dataset contains the smaller one, and a replay never moves from one split to the other.
    """
    replays = [r for r in db.replays_in_bucket(bucket) if r['pool_part'] not in (None, FAILED)]
    replays.sort(key=lambda r: pick_order(r, seed))
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


def build_dataset(
    db_path: Path,
    mds_dir: Path,
    n_train: int,
    n_val: int,
    seed: int,
    buckets: Sequence[str] = BUCKETS,
    slim: bool = False,
) -> None:
    """
    Copy the picked replays from the pools into `mds_dir/<bucket>/<split>`, if `slim` without
    the columns that the agents don't use.
    """
    columns = {
        name: dtype
        for name, dtype in MDS_DTYPE_STR_BY_COLUMN.items()
        if not (slim and name in UNUSED_COLUMNS)
    }
    db = DataBase(db_path)
    db.create_replays_table()
    for bucket in buckets:
        splits = pick_replays(db, BUCKETS.index(bucket), n_train, n_val, seed)
        # go through the pools part by part, as reading a part means unpacking its shards
        wanted: Dict[str, List[tuple[int, str]]] = defaultdict(list)
        for split, replays in splits.items():
            for r in replays:
                wanted[r['pool_part']].append((r['pool_pos'], split))
        writers = {
            split: MDSWriter(
                out=(mds_dir / bucket / split).as_posix(),
                columns=columns,
                compression='zstd',
                size_limit=1 << 31,
                exist_ok=True,
            )
            for split in splits
        }
        for part in tqdm(sorted(wanted), desc=f'Building {bucket}'):
            # unpack in a scratch directory, not in the pool
            with tempfile.TemporaryDirectory(dir=mds_dir, prefix='fax-build-') as cache:
                pool = StreamingDataset(
                    remote=(mds_dir / POOL_DIR / part).as_posix(),
                    local=cache,
                    shuffle=False,
                    batch_size=1,
                )
                for pos, split in sorted(wanted[part]):
                    sample = pool[pos]
                    writers[split].write({name: sample[name] for name in columns})
                del pool
        for split, writer in writers.items():
            writer.finish()
            logger.info(f'Wrote {len(splits[split])} replays to {mds_dir / bucket / split}')
    db.close()


if __name__ == '__main__':
    exposed_args = {
        'PATHS': 'sql mds',
        'BASE': 'seed debug',
        'TRAINING': 'n-samples n-val-samples',
    }
    parser = create_parser(exposed_args)
    parser.add_argument(
        '--buckets', nargs='+', choices=BUCKETS, default=list(BUCKETS), help='Buckets to build.'
    )
    parser.add_argument(
        '--slim',
        action='store_true',
        help='Leave out the columns the agents do not use (projectiles and ranks), 42%% smaller.',
    )
    args = parser.parse_args()
    cfg = parse_args(args, __file__)

    mds_dir = cfg.paths.mds.expanduser().resolve()
    for bucket in args.buckets:
        if (mds_dir / bucket).exists() and any((mds_dir / bucket).iterdir()):
            logger.error(f'{mds_dir / bucket} already has contents; please delete it first')
            sys.exit(1)

    build_dataset(
        cfg.paths.sql,
        mds_dir,
        n_train=cfg.training.n_samples,
        n_val=cfg.training.n_val_samples,
        seed=cfg.base.seed,
        buckets=args.buckets,
        slim=args.slim,
    )
