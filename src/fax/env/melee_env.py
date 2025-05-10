"""
SSBM environment wrapper for RL training.
"""

import time
from attr import define
from pathlib import Path

import numpy as np
import melee
from gymnasium import spaces

from fax.constants import PATHS


@define(auto_attribs=True, kw_only=True)
class MeleeEnv:
    exe_path: Path = PATHS.exe
    iso_path: Path = PATHS.iso
    replay_dir: Path = PATHS.replays
    headless: bool = True
    fast_forward: bool = True
    character: melee.Character = melee.Character.FOX
    opponent: melee.Character = melee.Character.FOX
    opponent_type: str = "cpu"
    opponent_level: int = 9
    stage: melee.Stage = melee.Stage.FINAL_DESTINATION
    console: melee.Console = None
    controller: melee.Controller = None
    opponent_controller: melee.Controller = None
    menu_helper: melee.MenuHelper = None
    last_frame: int = -999
    action_space: spaces.Dict = None
    observation_space: spaces.Dict = None

    def __attrs_post_init__(self):
        """Initialize the environment."""
        # Set up the replay directory
        self.replay_dir = self.replay_dir / f"{self.character}_{self.opponent}_{self.stage}"
        self.replay_dir.mkdir(parents=True, exist_ok=True)

        # Set up action and observation spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Set up action and observation spaces."""
        # Action space: main stick (x,y), c-stick (x,y), shoulder (L,R), and 8 buttons
        self.action_space = spaces.Dict(
            {
                "main_stick": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "c_stick": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "shoulder": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "buttons": spaces.MultiBinary(8),  # A, B, X, Y, Z, L, R, START
            }
        )

        # Observation space: player state, opponent state, stage info
        # This is a simplified version - we'll expand this later
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "position": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32),
                        "velocity": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32),
                        "percent": spaces.Box(low=0, high=999, shape=(1,), dtype=np.float32),
                        "facing": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                        "action": spaces.Discrete(386),  # Number of possible actions in Melee
                        "action_frame": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                        "jumps_left": spaces.Discrete(6),
                        "on_ground": spaces.Discrete(2),
                        "shield_strength": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
                        "stock": spaces.Discrete(5),
                    }
                ),
                "opponent": spaces.Dict(
                    {
                        "position": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32),
                        "velocity": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32),
                        "percent": spaces.Box(low=0, high=999, shape=(1,), dtype=np.float32),
                        "facing": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                        "action": spaces.Discrete(386),
                        "action_frame": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                        "jumps_left": spaces.Discrete(6),
                        "on_ground": spaces.Discrete(2),
                        "shield_strength": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
                        "stock": spaces.Discrete(5),
                    }
                ),
                "stage": spaces.Box(
                    low=-100, high=100, shape=(4,), dtype=np.float32
                ),  # stage boundaries
            }
        )

    def reset(self):
        """Reset the environment and return the initial observation."""
        # Clean up existing console if it exists
        if self.console:
            if self.controller:
                self.controller.disconnect()
            if self.opponent_controller:
                self.opponent_controller.disconnect()
            self.console.stop()
            time.sleep(0.5)  # Give the console time to clean up

        # Set up console and controllers
        self.console = melee.Console(
            path=str(self.exe_path),
            gfx_backend="Null" if self.headless else "",
            disable_audio=True,
            use_exi_inputs=True,
            enable_ffw=self.fast_forward,
            replay_dir=str(self.replay_dir),
        )

        self.controller = melee.Controller(console=self.console, port=1)
        self.opponent_controller = melee.Controller(console=self.console, port=2)

        self.console.run(iso_path=str(self.iso_path))
        if not self.console.connect():
            raise RuntimeError("Failed to connect to console")

        if not self.controller.connect() or not self.opponent_controller.connect():
            raise RuntimeError("Failed to connect controllers")

        self.menu_helper = melee.MenuHelper()
        self.last_frame = -999

        # Navigate menus to start a game
        gamestate = None
        while gamestate is None or gamestate.menu_state != melee.Menu.IN_GAME:
            gamestate = self.console.step()
            if gamestate is None:
                continue

            self.menu_helper.menu_helper_simple(
                gamestate,
                self.controller,
                self.character,
                self.stage,
                "",
                self.opponent_level if self.opponent_type == "cpu" else 0,
                1,
                True,
            )
            self.menu_helper.menu_helper_simple(
                gamestate,
                self.opponent_controller,
                self.opponent,
                self.stage,
                "",
                0,
                2,
                False,
            )

        # Return initial observation
        return self._get_observation(gamestate)

    def step(self, action):
        """Take a step in the environment."""
        if self.console is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Apply action to controller
        self._apply_action(action)

        # Step the environment
        gamestate = self.console.step()
        while gamestate is None:
            gamestate = self.console.step()

        # Check if game is over
        done = False
        if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            done = True
        elif gamestate.frame < self.last_frame:
            done = True

        self.last_frame = gamestate.frame

        # Calculate reward
        reward = self._calculate_reward(gamestate)

        # Get observation
        observation = self._get_observation(gamestate)

        # Additional info
        info = {
            "frame": gamestate.frame,
        }

        return observation, reward, done, info

    def _apply_action(self, action):
        """Apply the action to the controller."""
        # Main stick
        self.controller.tilt_analog(
            melee.Button.BUTTON_MAIN, action["main_stick"][0], action["main_stick"][1]
        )

        # C-stick
        self.controller.tilt_analog(
            melee.Button.BUTTON_C, action["c_stick"][0], action["c_stick"][1]
        )

        # Shoulder buttons
        self.controller.press_shoulder(melee.Button.BUTTON_L, action["shoulder"][0])
        self.controller.press_shoulder(melee.Button.BUTTON_R, action["shoulder"][1])

        # Buttons
        buttons = [
            melee.Button.BUTTON_A,
            melee.Button.BUTTON_B,
            melee.Button.BUTTON_X,
            melee.Button.BUTTON_Y,
            melee.Button.BUTTON_Z,
            melee.Button.BUTTON_L,
            melee.Button.BUTTON_R,
            melee.Button.BUTTON_START,
        ]

        for i, button in enumerate(buttons):
            if action["buttons"][i]:
                self.controller.press_button(button)
            else:
                self.controller.release_button(button)

        # Flush the controller
        self.controller.flush()

    def _get_observation(self, gamestate):
        """Convert gamestate to observation."""
        player = gamestate.players[1]  # Port 1
        opponent = gamestate.players[2]  # Port 2

        observation = {
            "player": {
                "position": np.array([player.position.x, player.position.y], dtype=np.float32),
                "velocity": np.array(
                    [player.speed_air_x_self, player.speed_y_self], dtype=np.float32
                ),
                "percent": np.array([player.percent], dtype=np.float32),
                "facing": np.array([1 if player.facing else -1], dtype=np.float32),
                "action": player.action.value,
                "action_frame": np.array([player.action_frame], dtype=np.float32),
                "jumps_left": player.jumps_left,
                "on_ground": int(player.on_ground),
                "shield_strength": np.array([player.shield_strength], dtype=np.float32),
                "stock": player.stock,
            },
            "opponent": {
                "position": np.array([opponent.position.x, opponent.position.y], dtype=np.float32),
                "velocity": np.array(
                    [opponent.speed_air_x_self, opponent.speed_y_self], dtype=np.float32
                ),
                "percent": np.array([opponent.percent], dtype=np.float32),
                "facing": np.array([1 if opponent.facing else -1], dtype=np.float32),
                "action": opponent.action.value,
                "action_frame": np.array([opponent.action_frame], dtype=np.float32),
                "jumps_left": opponent.jumps_left,
                "on_ground": int(opponent.on_ground),
                "shield_strength": np.array([opponent.shield_strength], dtype=np.float32),
                "stock": opponent.stock,
            },
            "stage": np.array(
                [
                    gamestate.stage.blast_zone_left,
                    gamestate.stage.blast_zone_right,
                    gamestate.stage.blast_zone_top,
                    gamestate.stage.blast_zone_bottom,
                ],
                dtype=np.float32,
            ),
        }

        return observation

    def _calculate_reward(self, gamestate):
        """Calculate the reward based on the gamestate."""
        # Simple reward: damage dealt - damage taken + stock differential
        player = gamestate.players[1]
        opponent = gamestate.players[2]

        # Stock differential (weighted heavily)
        stock_diff = (player.stock - opponent.stock) * 10

        # Percent differential (negative because lower percent is better)
        percent_diff = (opponent.percent - player.percent) / 100

        # Combine rewards
        reward = stock_diff + percent_diff

        return reward

    def close(self):
        """Clean up resources."""
        if self.console:
            if self.controller:
                self.controller.disconnect()
            if self.opponent_controller:
                self.opponent_controller.disconnect()
            self.console.stop()
