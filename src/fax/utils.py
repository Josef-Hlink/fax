import random
import torch
from tensordict import TensorDict


def generate_random_inputs():
    return {
        'main_stick': (random.random(), random.random()),
        'c_stick': (random.random(), random.random()),
        'shoulder': 0,
        'buttons': {
            'BUTTON_A': random.randint(0, 1),
            'BUTTON_B': random.randint(0, 1),
            'BUTTON_X': random.randint(0, 1),
            'BUTTON_Y': random.randint(0, 1),
            'BUTTON_Z': random.randint(0, 1),
            'BUTTON_L': random.randint(0, 1),
            'BUTTON_R': random.randint(0, 1),
        },
    }


def flatten_tensordict(td: TensorDict) -> torch.Tensor:
    """
    Flattens a tensordict into a 1D torch tensor.
    """
    features = [
        'character',
        'action',
        'percent',
        'stock',
        'facing',
        'invulnerable',
        'jumps_left',
        'on_ground',
        'shield_strength',
        'position_x',
        'position_y',
    ]
    return torch.Tensor(
        [td[f'p1_{feature}'] for feature in features]
        + [td[f'p2_{feature}'] for feature in features]
    )


def controller_output_from_model(output: dict) -> dict:
    # Clamp or threshold as needed
    buttons = output['buttons'] > 0.5  # Convert to binary decisions
    buttons = buttons.squeeze(0).int().tolist()

    return {
        'main_stick': tuple(output['main_stick'].squeeze(0).tolist()),
        'c_stick': tuple(output['c_stick'].squeeze(0).tolist()),
        'shoulder': float(output['shoulder'].item()),
        'buttons': {
            'BUTTON_A': buttons[0],
            'BUTTON_B': buttons[1],
            'BUTTON_X': buttons[2],
            'BUTTON_Y': buttons[3],
            'BUTTON_Z': buttons[4],
            'BUTTON_L': buttons[5],
            'BUTTON_R': buttons[6],
        },
    }


def compute_reward(td: TensorDict) -> float:
    p1_x = float(td['p1_position_x'])
    p1_y = float(td['p1_position_y'])
    p2_x = float(td['p2_position_x'])
    p2_y = float(td['p2_position_y'])

    dx = p2_x - p1_x
    dy = p2_y - p1_y
    distance = (dx**2 + dy**2) ** 0.5

    return -distance  # Reward is higher when closer
