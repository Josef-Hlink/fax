# -*- coding: utf-8 -*-

"""Small miscellaneous utility functions."""

import sys
import time
from functools import wraps
from pathlib import Path

from loguru import logger


def timed(func):
    @wraps(func)
    def functimer(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f'{func.__name__} executed in {elapsed:.5f}s')
        return result

    return functimer
