#!/usr/bin/env python3

from pathlib import Path


_HOME = Path('~').expanduser().resolve()
_ROOT = _HOME / 'Developer' / 'own' / 'fax'
_DATA = _HOME / 'Data'

ISO = _DATA / 'Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso'
EXE = _DATA / 'Slippi_Online-x86_64-ExiAI.AppImage'  # headless ffw build
DOLPHIN_HOME = _HOME / '.config' / 'SlippiPlayback'
REPLAYS = _DATA / 'replays'

if not REPLAYS.exists():
    REPLAYS.mkdir(parents=True)
