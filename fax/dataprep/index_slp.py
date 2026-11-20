#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this script to index your .zip archives of .slp files into a database and
organize them into buckets based on the number of fox players in each replay.
This is a prerequisite for converting .slp files into an MDS dataset.
"""

import gzip
import shutil
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

from loguru import logger
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.dataprep.database import DataBase
from fax.dataprep.slp_reader import parse_train_replay


BUCKETS = ('nofox', 'onefox', 'twofox')


def process_zip_arch(
    arch_file: Path,
    out_dir: Path,
    db_path: Path,
    bucket_limit: int,
    buckets: Sequence[str] = BUCKETS,
) -> None:
    """Process a .zip archive of .slp files, indexing them into a database and
    organizing them into buckets based on the number of fox players.
    Args:
        arch_file: Path to the .zip archive.
        out_dir: Directory to store the organized .slp files: nofox, onefox, twofox.
        db_path: Path to the SQLite database file.
        bucket_limit: Maximum number of files per bucket.
        buckets: The buckets to fill; the others are left as they are.
    """
    # make a temporary directory for extraction right next to archive
    # so we're sure it's on the same disk (named so we never wipe someone's unrelated 'tmp')
    if (tmp := arch_file.parent / 'fax-index-tmp').exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    # make sure output directories exist
    dirs = tuple(Path(out_dir) / bucket for bucket in BUCKETS)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    db = DataBase(db_path)
    db.create_replays_table()
    # replays that an earlier run already indexed are skipped, so a run can add to an existing index
    indexed = {f.name for d in dirs for f in d.iterdir()}
    # a bucket we are not filling counts as full
    bucket_counts = [len(list(d.iterdir())) if d.name in buckets else bucket_limit for d in dirs]

    logger.info(f'Indexing .slp files in {arch_file}...')
    with ZipFile(arch_file, 'r') as zip_arch:
        iterator = tqdm(zip_arch.infolist(), desc=f'Indexing {arch_file.name}')
        for i, member in enumerate(iterator):
            # periodically check if all buckets are full, otherwise update tqdm
            if i % 100 == 0:
                if all(count >= bucket_limit for count in bucket_counts):
                    logger.info(f'All buckets are full: ({bucket_limit} files). Stopping indexing.')
                    break
                iterator.set_postfix({str(i): c for i, c in enumerate(bucket_counts)})

            # skip non-slp files
            raw_name = Path(member.filename).name
            if not raw_name.endswith('.slp') and not raw_name.endswith('.slp.gz'):
                logger.debug(f'Skipping non-slp file {raw_name}')
                continue
            if raw_name.removesuffix('.gz') in indexed:
                continue

            # extract file to tmp directory
            tmp_path = tmp / raw_name
            with zip_arch.open(member) as src, open(tmp_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            file = maybe_gunzip(tmp_path)

            # parse and index the replay
            record = parse_train_replay(file, arch_file.name, bucket_counts, bucket_limit)
            if record is not None:
                db.insert_replay(record)
                dest = dirs[record.bucket] / record.file_name
                shutil.move(file, dest)
                bucket_counts[record.bucket] += 1
            else:
                file.unlink()
    shutil.rmtree(tmp)
    db.close()
    logger.info(f'Indexing of {arch_file} completed.')


def maybe_gunzip(path: Path) -> Path:
    """If the given path ends with .gz, extract it and return the new path.
    Otherwise, return the original path.
    Any .gz file is deleted after extraction.
    """
    if path.suffix != '.gz':
        return path
    unz_path = path.with_suffix('')
    with gzip.open(path, 'rb') as f_in, open(unz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    path.unlink()
    return unz_path


if __name__ == '__main__':
    exposed_args = {
        'PATHS': 'zips slp sql',
        'BASE': 'debug',
        'TRAINING': 'n-samples n-val-samples',
    }
    parser = create_parser(exposed_args)
    parser.add_argument(
        '--buckets',
        nargs='+',
        choices=BUCKETS,
        default=list(BUCKETS),
        help='Buckets to fill. With an existing index, the run adds to it.',
    )
    args = parser.parse_args()
    cfg = parse_args(args, __file__)

    if cfg.paths.sql.exists():
        logger.info(f'Database file {cfg.paths.sql} already exists, adding to it.')

    for arch_file in sorted(cfg.paths.zips, key=lambda p: p.name):
        if not arch_file.is_file() or not arch_file.name.endswith('.zip'):
            logger.warning(f'Skipping non-zip file {arch_file}')
            continue
        process_zip_arch(
            arch_file,
            out_dir=cfg.paths.slp,
            db_path=cfg.paths.sql,
            bucket_limit=cfg.training.n_samples + cfg.training.n_val_samples,
            buckets=args.buckets,
        )

    logger.info('All done! You can now proceed to convert the .slp files into MDS format.')
