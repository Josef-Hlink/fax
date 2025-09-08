# -*- coding: utf-8 -*-

"""Small miscellaneous utility functions."""

import sys
import time
from pathlib import Path
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


def setup_logger(path: Path, debug: bool = False, suppress_stderr: bool = False) -> None:
    """Set up the logger to log to a file and optionally to stderr.
    Args:
        path: Path to the log file.
        debug: If True, set log level to DEBUG, else INFO.
        suppress_stderr: If True, do not log to stderr.
    """
    logger.remove()  # remove default logger
    logger.add(path, level='TRACE', enqueue=True)  # always log to file
    if suppress_stderr:
        return
    logger.add(sys.stderr, level='DEBUG' if debug else 'INFO')
    logger.debug('Debug to stderr enabled')
    return


def debug_enabled() -> bool:
    return logger.level('DEBUG').no >= logger._core.min_level  # type: ignore[attr-defined]
