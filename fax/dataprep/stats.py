#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature statistics calculation, saving, and loading for MDS datasets.
Callable as script for calculating and saving stats to JSON.

Largely adapted from https://github.com/ericyuegu/hal
"""

import json
from pathlib import Path
from typing import Dict

import attr
import numpy as np
import numpy.ma as ma
from loguru import logger
from streaming import Stream, StreamingDataset
from tqdm import tqdm

from fax.config import create_parser, parse_args
from fax.constants import NP_MASK_VALUE


@attr.s(auto_attribs=True, frozen=True)
class FeatureStats:
    """Contains mean, std, min, and max for a feature."""

    mean: float
    std: float
    min: float
    max: float


def load_dataset_stats(path: Path) -> Dict[str, FeatureStats]:
    """Load the dataset statistics from a JSON file.
    Args:
        path (Path): Path to the directory containing the stats.json file.
    Returns:
        Dict[str, FeatureStats]: A dictionary mapping feature names to their statistics.
    """
    with open(Path(path) / 'stats.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {feature: FeatureStats(**stats) for feature, stats in data.items()}


def calculate_dataset_stats(path: Path) -> None:
    """Calculate and save statistics for each feature in the MDS dataset to a JSON.
    Args:
        path (Path): Path of the MDS dataset directory (either train or val).
    Returns:
        None, saves stats to stats.json in the given directory (alongside index.json).
    """
    dataset = StreamingDataset(streams=[Stream(local=path.as_posix())])
    statistics = {}
    logger.info(f'Starting statistics calculation for dataset at {path}')

    for example in tqdm(dataset, total=dataset.size, desc='Calculating statistics'):
        for field_name, field_data in example.items():
            if field_name not in statistics:
                statistics[field_name] = {
                    'count': 0,
                    'mean': 0,
                    'M2': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                }

            numpy_array = ma.masked_greater_equal(field_data, NP_MASK_VALUE)

            valid_data = numpy_array.compressed()  # get only non-masked values

            feature_stats = statistics[field_name]
            feature_stats['count'] += valid_data.size
            delta = valid_data - feature_stats['mean']
            feature_stats['mean'] += np.sum(delta) / feature_stats['count']
            delta2 = valid_data - feature_stats['mean']
            feature_stats['M2'] += np.sum(delta * delta2)
            feature_stats['min'] = min(feature_stats['min'], np.min(valid_data))
            feature_stats['max'] = max(feature_stats['max'], np.max(valid_data))

    for field_name, feature_stats in statistics.items():
        if feature_stats['count'] > 0:
            feature_stats['std'] = np.sqrt(feature_stats['M2'] / feature_stats['count'])
        else:
            logger.warning(f'No valid numeric data for {field_name}')

        # we're only interested in mean, std, min, max
        del feature_stats['count']
        del feature_stats['M2']

        for key, value in feature_stats.items():
            if isinstance(value, np.number):
                feature_stats[key] = value.item()

    logger.info(f'Saving statistics to {path / "stats.json"}')
    with open(path / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(statistics, f)

    logger.info('Statistics calculation completed')


if __name__ == '__main__':
    parser = create_parser({'PATH': 'mds', 'BASE': 'debug'})
    cfg = parse_args(parser.parse_args(), __file__)

    mds_path = cfg.paths.mds
    for bucket in ['nofox', 'onefox', 'twofox']:
        for split in ['train', 'val']:
            calculate_dataset_stats(mds_path / bucket / split)
            dataset_stats = load_dataset_stats(mds_path / bucket / split)
            logger.debug(f'Stats for {bucket}/{split}:')
            for feature, stats in dataset_stats.items():
                logger.debug(f'Stats for feature: {feature}')
                logger.debug(
                    f'    mean: {stats.mean:.4f}, std: {stats.std:.4f}, '
                    + f'min: {stats.min:.4f}, max: {stats.max:.4f}'
                )
