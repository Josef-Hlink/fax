#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this script to index your .zip archives of .slp files into a database.
Every replay that is usable for training gets a row, with its matchup bucket (the number of
fox players), its rank and where it is inside the archive. Nothing is extracted: converting
the replays into an MDS dataset reads them straight from the archives.
"""

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from zipfile import ZipFile

from loguru import logger
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.dataprep.database import DataBase
from fax.dataprep.slp_reader import TrainReplayRecord, extract_member, parse_train_replay


BUCKETS = ('nofox', 'onefox', 'twofox')

# the archive of this worker process
_zip_arch: Optional[ZipFile] = None


def scratch_dir() -> Path:
    """Where to put a replay for the moment it is being read; in memory if possible."""
    shm = Path('/dev/shm')
    return shm if shm.is_dir() else Path(tempfile.gettempdir())


def _open_archive(arch_file: Path) -> None:
    global _zip_arch
    _zip_arch = ZipFile(arch_file, 'r')


def _index_member(member: str) -> Tuple[str, Optional[TrainReplayRecord], Optional[str]]:
    assert _zip_arch is not None
    tmp_path = scratch_dir() / f'fax-index-{os.getpid()}.slp'
    try:
        extract_member(_zip_arch, member, tmp_path)
        record, reason = parse_train_replay(tmp_path, Path(_zip_arch.filename or '').name, member)
    # a broken member must not stop the run; peppi's panics are BaseExceptions, and a worker that
    # dies on one loses its chunk and leaves the pool waiting forever
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        record, reason = None, f'unreadable: {type(e).__name__}: {e}'
    finally:
        tmp_path.unlink(missing_ok=True)
    return member, record, reason


def index_zip_arch(arch_file: Path, db_path: Path, n_workers: int) -> None:
    """Index every .slp file in a .zip archive that is not in the database yet."""
    db = DataBase(db_path)
    db.create_replays_table()
    known = db.known_members(arch_file.name)
    with ZipFile(arch_file, 'r') as zip_arch:
        members = [
            name
            for name in zip_arch.namelist()
            if name.endswith(('.slp', '.slp.gz')) and name not in known
        ]
    logger.info(f'{arch_file.name}: {len(members)} replays to index, {len(known)} known already')

    records, skipped = [], []
    n_indexed = 0
    with mp.Pool(n_workers, initializer=_open_archive, initargs=(arch_file,)) as pool:
        results = pool.imap_unordered(_index_member, members, chunksize=16)
        for member, record, reason in tqdm(
            results, total=len(members), desc=f'Indexing {arch_file.name}'
        ):
            if record is not None:
                records.append(record)
            else:
                skipped.append((arch_file.name, member, reason or 'unknown'))
            if len(records) + len(skipped) >= 2000:
                n_indexed += len(records)
                db.insert_replays(records)
                db.insert_skipped(skipped)
                records, skipped = [], []
    n_indexed += len(records)
    db.insert_replays(records)
    db.insert_skipped(skipped)
    db.close()
    logger.info(f'{arch_file.name}: indexed {n_indexed} of {len(members)} replays')


if __name__ == '__main__':
    exposed_args = {'PATHS': 'zips sql', 'BASE': 'debug'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)

    if cfg.paths.sql.exists():
        logger.info(f'Database file {cfg.paths.sql} already exists, adding to it.')

    for arch_file in sorted(cfg.paths.zips, key=lambda p: p.name):
        if not arch_file.is_file() or not arch_file.name.endswith('.zip'):
            logger.warning(f'Skipping non-zip file {arch_file}')
            continue
        index_zip_arch(arch_file, cfg.paths.sql, n_workers=max(1, mp.cpu_count() - 4))

    db = DataBase(cfg.paths.sql)
    for bucket, name in enumerate(BUCKETS):
        logger.info(f'{name}: {len(db.replays_in_bucket(bucket))} replays')
    db.close()
    logger.info('All done! You can now proceed to convert the replays into MDS format.')
