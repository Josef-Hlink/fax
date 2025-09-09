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


_DEBUG_ENABLED = False


def setup_logger(path: Path, debug: bool = False, suppress_stderr: bool = False) -> None:
    """Set up the logger to log to a file and optionally to stderr.
    Args:
        path: Path to the log file.
        debug: If True, set log level to DEBUG, else INFO.
        suppress_stderr: If True, do not log to stderr.
    """
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = debug
    logger.remove()  # remove default logger
    logger.add(path, level='TRACE', enqueue=True)  # always log to file
    if suppress_stderr:
        return
    logger.add(sys.stderr, level='DEBUG' if debug else 'INFO')
    logger.debug('Debug to stderr enabled')
    return


def debug_enabled() -> bool:
    return _DEBUG_ENABLED
