#!/usr/bin/env python3
# minimal standalone example of how to use libmelee with headless slippi
# uv run example.py -e ~/Vault/fastslp.app -i ~/Vault/melee.iso -r ~/Desktop/

import argparse
import signal
import sys

import melee


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', '-e', type=str, help='slippi executable (.app)')
    parser.add_argument('--iso', '-i', type=str, help='melee iso')
    parser.add_argument('--rep', '-r', type=str, help='where to save replays')
    args = parser.parse_args()

    console = melee.Console(
        path=args.exe,
        gfx_backend='Null',
        disable_audio=True,
        use_exi_inputs=True,
        enable_ffw=True,
        replay_dir=args.rep,
    )

    contr1 = melee.Controller(console=console, port=1)
    contr2 = melee.Controller(console=console, port=2)

    def signal_handler(sig, frame):
        contr1.disconnect()
        contr2.disconnect()
        console.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    console.run(iso_path=args.iso)
    if not console.connect():
        sys.exit(1)
    if not contr1.connect():
        sys.exit(1)
    if not contr2.connect():
        sys.exit(1)

    menu_helper = melee.MenuHelper()
    fox, yoshis = melee.Character.FOX, melee.Stage.YOSHIS_STORY
    last_frame = -999
    while True:
        gs = console.step()  # gs = game state
        if gs is None:
            continue
        if gs.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            if gs.frame < last_frame:
                print('game over, exiting loop')
                console.stop()
                sys.exit(0)
            last_frame = gs.frame
            print(f'\r{gs.frame}', end='')
            # CONTROLLER MANIPULATION GOES HERE
            contr1.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0.5)  # walk right
            contr2.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 0.5)  # walk left
        else:  # we're in menus
            menu_helper.menu_helper_simple(gs, contr1, fox, yoshis, '', 0, 1, True)
            menu_helper.menu_helper_simple(gs, contr2, fox, yoshis, '', 0, 2, False)


if __name__ == '__main__':
    main()
