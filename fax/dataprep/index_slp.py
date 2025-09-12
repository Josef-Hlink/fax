#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this script to index a directory of .slp files into an SQLite database.
This is a prerequisite for converting .slp files into an MDS dataset.
"""

import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from fax.config import create_parser, debug_enabled, parse_args
from fax.dataprep.database import DataBase
from fax.slp_reader import parse_replay


def index_slp(slp_dir: Path, db_path: Path) -> None:
    """Index .slp files in the given directory into an SQLite database.
    Args:
        slp_dir: Directory containing .slp files to index.
        db_path: Path to the SQLite database file to create.
    """
    # create database and tables
    db = DataBase(db_path)
    logger.info(f'Created database at {db_path}')
    db.create_replays_table()
    db.create_errors_table()
    logger.info(f'Indexing .slp files in {slp_dir}...')

    # create iterator over .slp files
    iterator = slp_dir.rglob('*.slp')
    # if not in debug mode, wrap iterator in tqdm for sleeker UI
    if not debug_enabled():
        total = len(list(slp_dir.rglob('*.slp')))
        iterator = tqdm(iterator, desc=f'Indexing .slp files in {slp_dir}', total=total)

    # actually parse and store each file
    for slp_path in iterator:
        try:
            db.insert_replay(parse_replay(slp_path, parse_ranks=True, parse_full=True))
        except Exception as e:
            db.insert_error(str(slp_path.name), str(e))
    logger.info(f'Finished indexing .slp files into {db.db_path}')
    logger.info(f'{db.n_replays} replays successfully indexed')
    logger.info(f'{db.n_errors} files failed to parse')
    return


if __name__ == '__main__':
    exposed_args = {'PATHS': 'slp sql', 'BASE': 'debug'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)

    if not cfg.paths.slp.exists():
        raise FileNotFoundError(f'.slp directory {cfg.paths.slp} does not exist')
    if cfg.paths.sql.exists():
        logger.error(f'Database file {cfg.paths.sql} already exists. Please delete it first.')
        sys.exit(1)

    index_slp(cfg.paths.slp, cfg.paths.sql)
