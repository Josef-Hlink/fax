import random


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
