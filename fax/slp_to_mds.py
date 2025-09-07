#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a directory of .slp files into an MDS dataset.
"""

from itertools import islice
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from fax.utils import debug_enabled
from fax.database import DataBase
from fax.slp_reader import parse_replay


def main(slp_dir: Path, db_path: Path, n: int = -1) -> None:
    db = setup_database(db_path)
    parse_and_store(slp_dir, db, n)


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
            db.insert_replay(parse_replay(slp_path, parse_ranks=True))
        except Exception as e:
            db.insert_error(str(slp_path.name), str(e))

    logger.info(f'Finished indexing .slp files into {db.db_path}')
    logger.info(f'{db.n_replays} replays successfully indexed')
    logger.info(f'{db.n_errors} files failed to parse')

    return


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from fax.paths import PROJECT_ROOT

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
        '-n',
        type=int,
        default=-1,
        help='Number of files to process. Default is -1 (process all files).',
    )
    parser.add_argument(
        '--db_path',
        type=Path,
        default=PROJECT_ROOT / 'db.sqlite',
        help='Path where the SQLite database will be created. '
        + 'Defaults to <project_root>/db.sqlite.',
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

    # set up logging
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if args.debug else 'INFO')
    logger.debug('Debug mode enabled')

    main(slp_dir, db_path, args.n)
