#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a directory of .slp files into an MDS dataset.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from fax.paths import PROJECT_ROOT
from fax.data_prep.database import parse_ranked_replays


def main(slp_dir: Path, db_path: Path, n: int = -1, debug: bool = False) -> None:
    parse_ranked_replays(slp_dir, db_path, n, debug)


if __name__ == '__main__':
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
        default=None,
        help='Path where the SQLite database will be created. '
        + 'Defaults to <project_root>/db.sqlite.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode.',
    )
    args = parser.parse_args()
    # resolve db_path
    if (db_path := args.db_path) is None:
        db_path = PROJECT_ROOT / 'db.sqlite'
    main(args.slp_dir, db_path, args.n, args.debug)
