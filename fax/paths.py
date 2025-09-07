#!/usr/bin/env python3

from pathlib import Path


_HOME = Path('~').expanduser().resolve()
_DATA = _HOME / 'Data'

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ISO = _DATA / 'Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso'
EXE = _DATA / 'Slippi_Online-x86_64-ExiAI.AppImage'  # headless ffw build
DOLPHIN_HOME = _HOME / '.config' / 'SlippiPlayback'
REPLAYS = _DATA / 'replays'

if not REPLAYS.exists():
    REPLAYS.mkdir(parents=True)
