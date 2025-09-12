#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
All path constants used in the project.
"""

import tomllib
from pathlib import Path

_PROJ = Path(__file__).parent.parent.resolve()

with open(_PROJ / 'config.toml', 'rb') as f:
    paths_config = tomllib.load(f)
ISO = Path(paths_config['paths']['iso']).expanduser().resolve()
EXE = Path(paths_config['paths']['exe']).expanduser().resolve()
SQL = Path(paths_config['paths']['sql']).expanduser().resolve()
MDS = Path(paths_config['paths']['mds']).expanduser().resolve()
RUNS = Path(paths_config['paths']['runs']).expanduser().resolve()
REPLAYS = Path(paths_config['paths']['replays']).expanduser().resolve()
DOLPHIN_HOME = Path(paths_config['paths']['dolphin-home']).expanduser().resolve()

if not RUNS.exists():
    RUNS.mkdir(parents=True)

LOG_DIR = _PROJ / 'logs'
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)


if __name__ == '__main__':
    print(f'Project root: {_PROJ}')
    print(f'ISO path: {ISO}')
    print(f'EXE path: {EXE}')
    print(f'SQL path: {SQL}')
    print(f'MDS path: {MDS}')
    print(f'Replay output directory: {RUNS}')
    print(f'Dolphin home: {DOLPHIN_HOME}')
    print(f'Log directory: {LOG_DIR}')
