#!/usr/bin/env python3

from pathlib import Path
from attr import define

@define(frozen=True)
class MeleePaths:
    iso: Path = Path.home() / 'Vault' / 'Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso'
    exe: Path = Path.home() / 'Vault' / 'Slippi_Online-x86_64-ExiAI.AppImage'
    replays: Path = Path.home() / 'Desktop'

# Global config instance
PATHS = MeleePaths()

