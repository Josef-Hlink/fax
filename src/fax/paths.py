#!/usr/bin/env python3

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / 'data'

ISO = DATA / 'Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso'
EXE = DATA / 'Slippi_Online-x86_64-ExiAI.AppImage'
REPLAYS = DATA / 'replays'

if not REPLAYS.exists():
    REPLAYS.mkdir(parents=True)
