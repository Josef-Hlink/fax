#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
All path constants used in the project.
"""

import tomllib
from pathlib import Path


_PROJ = Path(__file__).parent.parent.resolve()
_DATA = _PROJ / 'data'

with open(_PROJ / 'paths.toml', 'rb') as f:
    paths_config = tomllib.load(f)
ISO = Path(paths_config['paths']['ISO']).expanduser().resolve()
EXE = Path(paths_config['paths']['EXE']).expanduser().resolve()
DOLPHIN_HOME = Path(paths_config['paths']['DOLPHIN_HOME']).expanduser().resolve()
REPLAY_OUTPUT = Path(paths_config['paths']['REPLAY_OUTPUT']).expanduser().resolve()

if not REPLAY_OUTPUT.exists():
    REPLAY_OUTPUT.mkdir(parents=True)

LOG_DIR = _PROJ / 'logs'
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)

DEFAULT_DB_PATH = _DATA / 'index.db'
DEFAULT_MDS_DIR = _DATA / 'mds'


if __name__ == '__main__':
    print(f'Project root: {_PROJ}')
    print(f'Data directory: {_DATA}')
    print(f'ISO path: {ISO}')
    print(f'EXE path: {EXE}')
    print(f'Dolphin home: {DOLPHIN_HOME}')
    print(f'Replay output directory: {REPLAY_OUTPUT}')
    print(f'Log directory: {LOG_DIR}')
    print(f'Default DB path: {DEFAULT_DB_PATH}')
    print(f'Default MDS directory: {DEFAULT_MDS_DIR}')
