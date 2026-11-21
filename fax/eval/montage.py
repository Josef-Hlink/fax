#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cut the best highlights (see highlights.py) into one mp4.

Every clip is rendered by Slippi's playback Dolphin, which can dump the frames and the audio
of a part of a replay, on a virtual display: no desktop is needed. ffmpeg then adds a caption
to every clip and joins them.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

FIRST_FRAME = -123  # game frame number of a replay's first frame
MAX_CLIP_FRAMES = 20 * 60
QUOTA = {
    'zero-to-death': 4,
    'gimp': 4,
    'combo': 4,
    'edgeguard': 2,
    'quick kill': 1,
    'self-destruct': 2,
}
TITLES = {
    'zero-to-death': 'ZERO TO DEATH',
    'gimp': 'GIMP',
    'combo': 'COMBO',
    'edgeguard': 'EDGEGUARD',
    'quick kill': 'QUICK KILL',
    'self-destruct': 'OOPS',
}
# the colour of an agent's costume, by the matchup it trained on
COLOURS = {'FvF': 'white', 'FvX': 'orange', 'XvX': '0xB49BFF', 'XvF': '0x6FD46F'}

DOLPHIN_INI = """[Movie]
DumpFrames = True
DumpFramesSilent = True
[DSP]
DumpAudio = True
DumpAudioSilent = True
Backend = Null
[Core]
EmulationSpeed = 0.0
[Interface]
ConfirmStop = False
"""
GFX_INI = """[Settings]
AspectRatio = 6
InternalResolutionFrameDumps = True
EFBScale = 4
BitrateKbps = 25000
[Hardware]
VSync = False
"""


def pick(highlights: List[Dict[str, Any]], quota: Dict[str, int]) -> List[Dict[str, Any]]:
    """The best highlights of every kind, never two that overlap in the same game."""
    picked: List[Dict[str, Any]] = []
    for kind, n in quota.items():
        of_kind = sorted((h for h in highlights if h['kind'] == kind), key=lambda h: -h['score'])
        taken = 0
        for h in of_kind:
            if taken == n:
                break
            overlaps = any(
                p['file'] == h['file']
                and p['start_frame'] <= h['end_frame']
                and h['start_frame'] <= p['end_frame']
                for p in picked
            )
            # one clip per game keeps the montage varied
            if overlaps or any(p['file'] == h['file'] for p in picked):
                continue
            picked.append(h)
            taken += 1
    return picked


def player_names(file: Path, names: Optional[List[str]]) -> List[str]:
    """Names of player 1 and 2: as given, or else from a directory called `<p1>-v-<p2>`."""
    if names:
        return names
    for parent in file.parents:
        if '-v-' in parent.name:
            return parent.name.split('-v-', 1)
    return ['player 1', 'player 2']


def xvfb_command() -> List[str]:
    if shutil.which('xvfb-run'):
        return ['xvfb-run']
    return ['nix', 'shell', 'nixpkgs#xvfb-run', '-c', 'xvfb-run']


def render_clip(
    highlight: Dict[str, Any], out: Path, playback: Path, iso: Path, names: List[str], font: str
) -> bool:
    """Render one highlight to an mp4 with its caption. Returns whether that worked."""
    if not Path(highlight['file']).exists():
        logger.warning(f'{highlight["file"]} is gone')  # dolphin would wait for it forever
        return False
    start, end = highlight['start_frame'], highlight['end_frame']
    start = max(start, end - MAX_CLIP_FRAMES)
    with tempfile.TemporaryDirectory(prefix='fax-montage-') as tmp_name:
        tmp = Path(tmp_name)
        (tmp / 'user' / 'Config').mkdir(parents=True)
        (tmp / 'user' / 'Config' / 'Dolphin.ini').write_text(DOLPHIN_INI)
        (tmp / 'user' / 'Config' / 'GFX.ini').write_text(GFX_INI)
        queue = [
            {
                'path': highlight['file'],
                'startFrame': start + FIRST_FRAME,
                'endFrame': end + FIRST_FRAME,
            }
        ]
        comm = {
            'mode': 'queue',
            'replay': '',
            'isRealTimeMode': False,
            'outputOverlayFiles': False,
            'commandId': out.stem,
            'queue': queue,
        }
        (tmp / 'comm.json').write_text(json.dumps(comm))
        command = [
            *xvfb_command(),
            '-a',
            '-s',
            '-screen 0 1280x1024x24',
            playback.as_posix(),
            '-i',
            (tmp / 'comm.json').as_posix(),
            '-b',
            '-e',
            iso.as_posix(),
            '-u',
            (tmp / 'user').as_posix(),
            '--cout',
        ]
        # dolphin does not exit when the queue is done, it only says so
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if '[NO_GAME]' in line:
                    break
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()

        video = tmp / 'user' / 'Dump' / 'Frames' / 'framedump0.avi'
        audio = [
            f for f in (tmp / 'user' / 'Dump' / 'Audio').glob('*.wav') if f.stat().st_size > 10_000
        ]
        if not video.exists():
            logger.warning(f'No frames dumped for {out.name}')
            return False

        attacker = highlight['attacker']
        kind = highlight['kind']
        who = names[attacker]
        versus = names[1 - attacker]
        (tmp / 'title.txt').write_text(TITLES.get(kind, kind.upper()))
        # the names side by side, each in the colour of its costume: the same line is drawn
        # three times from the left, every time a shorter part of it
        solo = kind == 'self-destruct'
        (tmp / 'both.txt').write_text(who if solo else f'{who}  vs  {versus}')
        (tmp / 'upto_vs.txt').write_text(who if solo else f'{who}  vs')
        (tmp / 'who.txt').write_text(who)
        (tmp / 'text.txt').write_text(highlight['text'])
        colour = COLOURS.get(who.split()[0], 'white')
        versus_colour = COLOURS.get(versus.split()[0], 'white')

        def caption(name: str, size: int, colour: str, x: str, y: str) -> str:
            return (
                f"drawtext=fontfile='{font}':textfile='{tmp / name}':fontsize={size}"
                f':expansion=none:fontcolor={colour}:x={x}:y={y}'
            )

        # the captions go in a bar under the picture, the game's own display is in the way
        filters = ','.join(
            [
                'scale=1280:720',
                'pad=1280:820:0:0:black',
                caption('both.txt', 30, versus_colour, '28', '738'),
                caption('upto_vs.txt', 30, '0x909090', '28', '738'),
                caption('who.txt', 30, colour, '28', '738'),
                caption('title.txt', 30, 'white', 'w-tw-28', '738'),
                caption('text.txt', 24, '0xD0D0D0', '28', '782'),
            ]
        )
        ffmpeg = ['ffmpeg', '-v', 'error', '-y', '-i', video.as_posix()]
        for f in audio:
            ffmpeg += ['-i', f.as_posix()]
        # one filter graph for both: ffmpeg does not take -vf next to -filter_complex
        graph = f'[0:v]{filters}[v]'
        if audio:
            mix = ''.join(f'[{i + 1}:a]' for i in range(len(audio)))
            graph += f';{mix}amix=inputs={len(audio)}:normalize=0[a]'
        ffmpeg += ['-filter_complex', graph, '-map', '[v]']
        ffmpeg += ['-map', '[a]'] if audio else ['-an']
        ffmpeg += ['-c:v', 'libx264', '-crf', '20', '-preset', 'fast', '-pix_fmt', 'yuv420p']
        ffmpeg += ['-r', '60', '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-shortest']
        ffmpeg.append(out.as_posix())
        # the dump's last frame is often damaged, which ffmpeg reports but gets past
        result = subprocess.run(ffmpeg, capture_output=True, text=True)
        if not out.exists() or out.stat().st_size == 0:
            logger.warning(f'ffmpeg failed for {out.name}: {result.stderr[:600]}')
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('highlights', type=Path, help='JSON file written by highlights.py.')
    parser.add_argument('out', type=Path, help='The mp4 to write.')
    parser.add_argument('--iso', type=Path, required=True, help='The Melee ISO.')
    parser.add_argument(
        '--playback',
        type=Path,
        default=Path.home() / '.config/Slippi Launcher/playback/Slippi_Playback-x86_64.AppImage',
        help="Slippi's playback Dolphin.",
    )
    parser.add_argument('--names', nargs=2, default=None, help='Names of player 1 and 2.')
    parser.add_argument('--n-workers', type=int, default=3, help='Clips to render at once.')
    parser.add_argument('--font', default=None, help='Font file for the captions.')
    args = parser.parse_args()

    font = (
        args.font
        or subprocess.run(
            ['fc-match', '-f', '%{file}', 'sans:bold'], capture_output=True, text=True
        ).stdout
    )
    picked = pick(json.loads(args.highlights.read_text()), QUOTA)
    logger.info(f'Rendering {len(picked)} clips')
    clip_dir = args.out.with_suffix('.clips')
    clip_dir.mkdir(parents=True, exist_ok=True)
    clips = [clip_dir / f'{i:02d}-{h["kind"].replace(" ", "-")}.mp4' for i, h in enumerate(picked)]

    def render(job: Any) -> bool:
        h, clip = job
        ok = render_clip(
            h, clip, args.playback, args.iso, player_names(Path(h['file']), args.names), font
        )
        logger.info(f'{clip.name}: {"done" if ok else "FAILED"} ({h["text"]})')
        return ok

    with ThreadPoolExecutor(args.n_workers) as pool:
        done = list(pool.map(render, zip(picked, clips)))
    # several dolphins starting at once sometimes leave one without a display: try those again
    done = [ok or render(job) for ok, job in zip(done, zip(picked, clips))]

    listing = clip_dir / 'clips.txt'
    listing.write_text(''.join(f"file '{c.as_posix()}'\n" for c, ok in zip(clips, done) if ok))
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing.as_posix()]
        + ['-c', 'copy', '-movflags', '+faststart', args.out.as_posix()],
        check=True,
    )
    logger.info(f'Wrote {args.out} with {sum(done)} clips')
