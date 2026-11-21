#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find the moments worth watching in a directory of replays: combos, zero-to-deaths, gimps,
edgeguards, quick kills and self-destructs. Writes them to a JSON file, best first, with the
frames to cut, for a montage or just to know where to look.

A combo is counted the way slippi-js does: it goes on for as long as the victim keeps getting
hit before having been free (not in hitstun, grabbed, down or teching) for 45 frames.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import attr
import numpy as np
from loguru import logger
from peppi_py import read_slippi

COMBO_RESET_FRAMES = 45
MULTI_HIT_FRAMES = 25
# x of the ledges, by slippi stage id
STAGE_EDGE = {2: 63.35, 3: 87.75, 8: 56.0, 28: 77.27, 31: 68.4, 32: 85.57}
MOVE_NAMES = {
    2: 'jab',
    3: 'jab',
    4: 'jab',
    5: 'rapid jab',
    6: 'dash attack',
    7: 'forward tilt',
    8: 'up tilt',
    9: 'down tilt',
    10: 'forward smash',
    11: 'up smash',
    12: 'down smash',
    13: 'nair',
    14: 'fair',
    15: 'bair',
    16: 'up air',
    17: 'dair',
    18: 'laser',
    19: 'side B',
    20: 'up B',
    21: 'shine',
    50: 'getup attack',
    51: 'getup attack',
    52: 'pummel',
    53: 'forward throw',
    54: 'back throw',
    55: 'up throw',
    56: 'down throw',
    61: 'ledge attack',
    62: 'ledge attack',
}


@attr.s(auto_attribs=True)
class Highlight:
    kind: str  # combo, zero-to-death, gimp, edgeguard, quick kill, self-destruct
    file: str
    attacker: int  # port index (0 or 1) of the player it is a highlight of
    start_frame: int  # index into the replay's frames
    end_frame: int
    score: float
    first_hit_frame: int = 0
    hits: int = 0
    damage: float = 0.0
    start_percent: float = 0.0
    end_percent: float = 0.0
    kill: bool = False
    moves: List[str] = attr.Factory(list)
    text: str = ''


def _player_arrays(port: Any) -> Dict[str, np.ndarray]:
    post = port.leader.post
    return {
        'percent': np.asarray(post.percent.to_numpy(), dtype=np.float32),
        'stocks': np.asarray(post.stocks.to_numpy(), dtype=np.int32),
        'state': np.asarray(post.state.to_numpy(), dtype=np.int32),
        'x': np.asarray(post.position.x.to_numpy(), dtype=np.float32),
        'y': np.asarray(post.position.y.to_numpy(), dtype=np.float32),
        'last_attack': np.asarray(post.last_attack_landed.to_numpy(), dtype=np.int32),
    }


def _compromised(state: np.ndarray) -> np.ndarray:
    """In hitstun, grabbed, lying down, teching or dying: not free to act."""
    return (
        ((state >= 0x4B) & (state <= 0x5B))  # damage
        | ((state >= 0xDF) & (state <= 0xE8))  # grabbed
        | ((state >= 0xB7) & (state <= 0xC6))  # down
        | ((state >= 0xC7) & (state <= 0xCC))  # tech
        | (state <= 0x0A)  # dying
    )


def find_highlights(file: Path) -> List[Highlight]:
    """All highlights of both players in one replay."""
    try:
        game = read_slippi(file.as_posix())
    except Exception as e:
        logger.debug(f'Skipping {file.name}: {e}')
        return []
    if game.frames is None or len(game.frames.ports) != 2:
        return []
    players = [_player_arrays(port) for port in game.frames.ports]
    n_frames = len(players[0]['percent'])
    edge = STAGE_EDGE.get(int(game.start.stage), 68.4)

    highlights: List[Highlight] = []
    for a, v in ((0, 1), (1, 0)):
        attacker, victim = players[a], players[v]
        hit_frames = np.flatnonzero(np.diff(victim['percent']) > 0) + 1
        death_frames = np.flatnonzero(np.diff(victim['stocks']) < 0) + 1
        compromised = _compromised(victim['state'])
        offstage = (np.abs(victim['x']) > edge) | (victim['y'] < -5)
        spawn_frames = np.concatenate([[0], death_frames])

        # group the hits into combos
        combos: List[List[int]] = []
        last_busy = -10_000
        for frame in hit_frames:
            if combos:
                busy = np.flatnonzero(compromised[combos[-1][-1] : frame]) + combos[-1][-1]
                last_busy = int(busy[-1]) if len(busy) else combos[-1][-1]
            if not combos or frame - last_busy > COMBO_RESET_FRAMES:
                combos.append([])
            combos[-1].append(int(frame))

        for hits in combos:
            start, last_hit = hits[0], hits[-1]
            # the victim dying before being free again makes it a kill
            deaths = death_frames[(death_frames > last_hit) & (death_frames <= last_hit + 300)]
            kill = False
            end = last_hit + 60
            if len(deaths):
                free = np.flatnonzero(~compromised[last_hit : deaths[0]])
                kill = len(free) <= COMBO_RESET_FRAMES or bool(offstage[last_hit])
                if kill:
                    end = int(deaths[0]) + 45
            start_percent = float(victim['percent'][start - 1])
            end_percent = float(victim['percent'][last_hit])
            damage = end_percent - start_percent
            # a throw or a multi-hit move adds damage several times; count it as one hit
            moves: List[str] = []
            previous = -MULTI_HIT_FRAMES
            for f in hits:
                move = MOVE_NAMES.get(int(attacker['last_attack'][f]), 'hit')
                if not (moves and move == moves[-1] and f - previous <= MULTI_HIT_FRAMES):
                    moves.append(move)
                previous = f
            n_hits = len(moves)
            since_spawn = start - int(spawn_frames[spawn_frames <= start][-1])
            hit_offstage = bool(offstage[last_hit])

            kind: Optional[str] = None
            score = damage + 6 * n_hits + (40 if kill else 0)
            if kill and start_percent < 1 and n_hits >= 3:
                kind, score = 'zero-to-death', score + 60
            elif kill and hit_offstage and end_percent < 60:
                kind, score = 'gimp', score + 70 - end_percent
            elif kill and hit_offstage and n_hits >= 2:
                kind, score = 'edgeguard', score + 25
            elif kill and since_spawn < 600 and start_percent < 1:
                kind, score = 'quick kill', score + 30
            elif n_hits >= 4 and damage >= 40:
                kind = 'combo'
            elif kill and n_hits >= 3 and damage >= 30:
                kind = 'combo'
            if kind is None:
                continue
            ending = f', kills at {end_percent:.0f}%' if kill else ''
            highlights.append(
                Highlight(
                    kind=kind,
                    file=file.as_posix(),
                    attacker=a,
                    start_frame=max(0, start - 150),
                    first_hit_frame=start,
                    end_frame=min(n_frames - 1, end),
                    score=round(float(score), 1),
                    hits=n_hits,
                    damage=round(damage, 1),
                    start_percent=round(start_percent, 1),
                    end_percent=round(end_percent, 1),
                    kill=kill,
                    moves=moves,
                    text=f'{n_hits} hits, {damage:.0f}%{ending}: ' + ', '.join(moves[-6:]),
                )
            )

        # deaths that nobody caused, for the blooper reel
        for death in death_frames:
            recent_hits = hit_frames[(hit_frames > death - 300) & (hit_frames < death)]
            if len(recent_hits) == 0:
                percent = float(victim['percent'][death - 1])
                highlights.append(
                    Highlight(
                        kind='self-destruct',
                        file=file.as_posix(),
                        attacker=v,
                        start_frame=max(0, int(death) - 240),
                        end_frame=min(n_frames - 1, int(death) + 45),
                        score=round(20 + max(0.0, 60 - percent), 1),
                        end_percent=round(percent, 1),
                        text=f'dies untouched at {percent:.0f}%',
                    )
                )
    return highlights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays', type=Path, help='Directory with .slp files (searched in depth).')
    parser.add_argument('--out', type=Path, default=None, help='JSON file to write to.')
    parser.add_argument('--top', type=int, default=20, help='How many to print per kind.')
    args = parser.parse_args()

    found: List[Highlight] = []
    files = sorted(args.replays.glob('**/*.slp'))
    for file in files:
        found.extend(find_highlights(file))
    found.sort(key=lambda h: -h.score)
    logger.info(f'{len(found)} highlights in {len(files)} replays')
    for kind in sorted({h.kind for h in found}):
        of_kind = [h for h in found if h.kind == kind]
        logger.info(f'{kind}: {len(of_kind)}')
        for h in of_kind[: args.top]:
            logger.info(
                f'  {h.score:6.1f}  p{h.attacker + 1}  {Path(h.file).name}  '
                f'frames {h.start_frame}-{h.end_frame}  {h.text}'
            )
    if args.out is not None:
        args.out.write_text(json.dumps([attr.asdict(h) for h in found], indent=1))
