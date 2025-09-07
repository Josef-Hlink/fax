#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a directory of .slp files into an MDS dataset.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from fax.data_prep.database import parse_ranked_replays


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
        '--debug',
        action='store_true',
        help='Enable debug mode.',
    )
    args = parser.parse_args()
    parse_ranked_replays(args.slp_dir, args.n, args.debug)
