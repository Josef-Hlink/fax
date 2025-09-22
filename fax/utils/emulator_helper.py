# -*- coding: utf-8 -*-

"""
Helper functions and classes used in evaluation steps that need to interact with the emulator.

Largely copied from https://github.com/ericyuegu/hal
"""

import concurrent.futures
import signal
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import attr
import melee
import psutil
from loguru import logger

from fax.utils.constants import ORIGINAL_BUTTONS, PLAYER_1_PORT, PLAYER_2_PORT, Player, get_opponent


def _get_console_port(player: Player) -> int:
    return PLAYER_1_PORT if player == 'p1' else PLAYER_2_PORT


def find_open_udp_ports(n: int, min_port=1024, max_port=65535) -> list[int]:
    """Only tested on Linux."""
    used_ports = set()
    for conn in psutil.net_connections(kind='udp'):
        if conn.laddr and isinstance(conn.laddr, tuple):
            used_ports.add(conn.laddr[1])  # laddr = (ip, port)
    open_ports = []
    for port in range(min_port, max_port):
        if port not in used_ports:
            open_ports.append(port)
        if len(open_ports) == n:
            break
    return open_ports


def get_headless_console_kwargs(
    replay_dir: Path,
    emulator_path: Path,
    enable_ffw: bool = True,
    udp_port: int | None = None,
    console_logger: melee.Logger | None = None,
) -> Dict[str, Any]:
    headless_console_kwargs = {
        'gfx_backend': 'Null',
        'disable_audio': True,
        'use_exi_inputs': enable_ffw,
        'enable_ffw': enable_ffw,
    }
    replay_dir.mkdir(exist_ok=True, parents=True)
    console_kwargs = {
        'path': emulator_path.as_posix(),
        'is_dolphin': True,
        'tmp_home_directory': True,
        'copy_home_directory': False,
        'replay_dir': replay_dir.as_posix(),
        'blocking_input': True,
        'slippi_port': udp_port,
        'online_delay': 0,  # 0 frame delay for local evaluation
        'logger': console_logger,
        **headless_console_kwargs,
    }
    return console_kwargs


def get_gui_console_kwargs(
    emulator_path: Path,
    replay_dir: Path,
    console_logger: melee.Logger | None = None,
) -> Dict[str, Any]:
    """Get console kwargs for GUI-enabled emulator."""
    replay_dir.mkdir(exist_ok=True, parents=True)
    console_kwargs = {
        'path': emulator_path.as_posix(),
        'is_dolphin': True,
        'tmp_home_directory': True,
        'copy_home_directory': False,
        'replay_dir': replay_dir.as_posix(),
        'blocking_input': False,
        'slippi_port': 51441,  # must use default port for local mainline/Ishiiruka
        'online_delay': 0,  # 0 frame delay for local evaluation
        'logger': console_logger,
        'setup_gecko_codes': True,
        'fullscreen': False,
        'gfx_backend': '',
        'disable_audio': False,
        'use_exi_inputs': False,
        'enable_ffw': False,
    }
    return console_kwargs


@attr.s(auto_attribs=True)
class MatchupMenuHelper:
    controller_1: melee.Controller
    controller_2: melee.Controller
    character_1: melee.Character
    character_2: Optional[melee.Character]
    stage: Optional[melee.Stage]
    opponent_cpu_level: int = 9

    # internal use
    _player_1_character_selected: bool = False

    def select_character_and_stage(self, gamestate: melee.GameState) -> None:
        """
        Call this helper function every frame to handle menu state logic.

        If character_2 or stage_selected is None, the function will wait for human user.
        """
        menu_helper = melee.menuhelper.MenuHelper()
        if gamestate.menu_state == melee.enums.Menu.MAIN_MENU:
            melee.menuhelper.MenuHelper.choose_versus_mode(
                gamestate=gamestate, controller=self.controller_1
            )
        # if we're at the character select screen, choose our character
        elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
            menu_helper.choose_character(
                character=self.character_1,
                gamestate=gamestate,
                controller=self.controller_1,
                cpu_level=0,  # human
                costume=0,
                swag=False,
                start=False,
            )
            if self.character_2 is None:
                return
            menu_helper.choose_character(
                character=self.character_2,
                gamestate=gamestate,
                controller=self.controller_2,
                cpu_level=self.opponent_cpu_level,
                costume=1,
                swag=False,
                start=True,
            )
        # if we're at the stage select screen, choose a stage
        elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
            if self.stage is None:
                return
            menu_helper.choose_stage(
                stage=self.stage,
                gamestate=gamestate,
                controller=self.controller_1,
                character=self.character_1,
            )
        # if we're at the postgame scores screen, spam START
        elif gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
            menu_helper.skip_postgame(controller=self.controller_1)


@contextmanager
def console_manager(console: melee.Console, console_logger: melee.Logger | None = None):
    def signal_handler(sig, frame):
        raise KeyboardInterrupt

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        yield
    except KeyboardInterrupt:
        logger.info('Received interrupt, shutting down...')
    except TimeoutError:
        raise
    except Exception as e:
        logger.error(
            f'Stopping console due to exception: {e}\nTraceback:\n{"".join(traceback.format_tb(e.__traceback__))}'
        )
        raise
    finally:
        if console_logger is not None:
            console_logger.writelog()
            logger.info('Log file created: ' + console_logger.filename)
        signal.signal(signal.SIGINT, original_handler)
        console.stop()
        logger.info('Shutting down cleanly...')


def send_controller_inputs(controller: melee.Controller, inputs: Dict[str, Any]) -> None:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller (melee.Controller): Controller object.
        inputs (Dict[str, Any]): Dictionary of controller inputs
    """
    controller.tilt_analog(
        melee.Button.BUTTON_MAIN,
        inputs['main_stick'][0],
        inputs['main_stick'][1],
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C,
        inputs['c_stick'][0],
        inputs['c_stick'][1],
    )
    # handle shoulder input from either format
    shoulder_value = inputs.get('shoulder', inputs.get('analog_shoulder', 0))
    controller.press_shoulder(
        melee.Button.BUTTON_L,
        shoulder_value,
    )

    buttons_to_press: List[str] = inputs.get('buttons', [])
    for button_str in ORIGINAL_BUTTONS:
        button = getattr(melee.Button, button_str.upper())
        if button_str in buttons_to_press:
            controller.press_button(button)
        else:
            controller.release_button(button)

    controller.flush()


@attr.s(auto_attribs=True)
class Matchup:
    stage: str = 'BATTLEFIELD'
    ego_character: str = 'FOX'
    opponent_character: str = 'FOX'


@attr.s(auto_attribs=True)
class EmulatorManager:
    udp_port: int
    player: Player
    emulator_path: Path
    replay_dir: Path
    opponent_cpu_level: int = 9
    matchup: Matchup = Matchup(stage='BATTLEFIELD', ego_character='FOX', opponent_character='FOX')
    max_steps: int = 99999
    latency_warning_threshold: float = 14.0
    console_timeout: float = 5.0
    enable_ffw: bool = True
    debug: bool = False

    def __attrs_post_init__(self) -> None:
        self.console_logger = melee.Logger() if self.debug else None
        console_kwargs = get_headless_console_kwargs(
            emulator_path=self.emulator_path,
            enable_ffw=self.enable_ffw,
            udp_port=self.udp_port,
            replay_dir=self.replay_dir,
            console_logger=self.console_logger,
        )
        self.console = melee.Console(**console_kwargs)
        self.ego_controller = melee.Controller(
            console=self.console,
            port=_get_console_port(self.player),
            type=melee.ControllerType.STANDARD,
        )
        self.opponent_controller = melee.Controller(
            console=self.console,
            port=_get_console_port(get_opponent(self.player)),
            type=melee.ControllerType.STANDARD,
        )
        self.menu_helper = MatchupMenuHelper(
            controller_1=self.ego_controller,
            controller_2=self.opponent_controller,
            character_1=melee.Character[self.matchup.ego_character],
            character_2=melee.Character[self.matchup.opponent_character],
            stage=melee.Stage[self.matchup.stage],
            opponent_cpu_level=self.opponent_cpu_level,
        )

    def run_game(
        self, iso_path: Path
    ) -> Generator[melee.GameState, Tuple[Dict[str, Any], Dict[str, Any] | None], None]:
        """Generator that yields gamestates and receives controller inputs.

        Yields:
            Optional[melee.GameState]: The current game state, or None if the episode is over

        Sends:
            TensorDict: Controller inputs to be applied to the game
        """
        # run the console
        self.console.run(iso_path=iso_path.as_posix())
        # connect to the console
        logger.debug('Connecting to console...')
        if not self.console.connect():
            logger.debug('ERROR: Failed to connect to the console.')
            sys.exit(-1)
        logger.debug('Console connected')

        # connect controllers
        logger.debug('Connecting controller 1 to console...')
        if not self.ego_controller.connect():
            logger.debug('ERROR: Failed to connect the controller.')
            sys.exit(-1)
        logger.debug('Controller 1 connected')
        logger.debug('Connecting controller 2 to console...')
        if not self.opponent_controller.connect():
            logger.debug('ERROR: Failed to connect the controller.')
            sys.exit(-1)
        logger.debug('Controller 2 connected')

        i = 0

        # this whole double with block is to ensure clean shutdown of the console and controller
        # i've battled with this for days and this is the only way i've found that works
        # reliable enough...
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
            console_manager(console=self.console, console_logger=self.console_logger),
        ):
            logger.debug(
                f'Starting episode on {self.matchup.stage}: {self.matchup.ego_character} vs. {self.matchup.opponent_character}'
            )
            while i < self.max_steps:
                # wrap `console.step()` in a thread with timeout
                future = executor.submit(self.console.step)
                try:
                    gamestate = future.result(timeout=self.console_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error('console.step() timed out')
                    raise

                if gamestate is None:
                    logger.debug('Gamestate is None')
                    continue

                if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    self.menu_helper.select_character_and_stage(gamestate)
                else:
                    # yield gamestate and receive controller inputs
                    controller_inputs = yield gamestate
                    if controller_inputs is None:
                        logger.debug('Controller inputs are None')
                    else:
                        ego_controller_inputs, opponent_controller_inputs = controller_inputs
                        send_controller_inputs(self.ego_controller, ego_controller_inputs)
                        if opponent_controller_inputs is not None:
                            send_controller_inputs(
                                self.opponent_controller, opponent_controller_inputs
                            )

                    if self.console_logger is not None:
                        self.console_logger.writeframe()
                    i += 1
