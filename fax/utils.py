# -*- coding: utf-8 -*-

"""Small miscellaneous utility functions."""

import time
from loguru import logger
from functools import wraps


def timed(func):
    @wraps(func)
    def functimer(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f'{func.__name__} executed in {elapsed:.5f}s')
        return result

    return functimer


def debug_enabled() -> bool:
    return logger.level('DEBUG').no >= logger._core.min_level  # type: ignore[attr-defined]
