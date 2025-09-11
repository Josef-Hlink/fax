# -*- coding: utf-8 -*-

import os
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

import attr
import wandb
import torch
from loguru import logger
from tensordict import TensorDict

from fax.config import Config


@attr.s(auto_attribs=True, slots=True)
class WandbConfig:
    project: str
    train_config: Dict[str, Any]
    tags: List[str]
    model: torch.nn.Module

    @classmethod
    def create(cls, model: torch.nn.Module, train_config: Config) -> Optional['WandbConfig']:
        if not os.getenv('WANDB_API_KEY'):
            logger.info('W&B run not initiated because WANDB_API_KEY not set.')
            return None
        if train_config.debug:
            logger.info('Debug mode, skipping W&B.')
            return None

        model_name = model.model.__class__.__name__
        config = {'model_name': model_name, **vars(train_config)}
        tags = [model_name]

        return cls(project='hal', train_config=config, tags=tags, model=model)


class Writer:
    def __init__(self, wandb_config: WandbConfig) -> None:
        self.wandb_config = wandb_config
        wandb.init(
            project=wandb_config.project,
            config=wandb_config.train_config,
            tags=wandb_config.tags,
        )
        train_config = wandb_config.train_config
        log_freq = train_config['report_len'] // (
            train_config['local_batch_size'] * train_config['n_gpus']
        )
        wandb.watch(wandb_config.model, log='all', log_freq=log_freq)

    def log(
        self, summary_dict: TensorDict | Dict[str, Any], step: int, commit: bool = True
    ) -> None:
        """Add on event to the event file."""
        if isinstance(summary_dict, TensorDict):
            summary_dict = {
                k: v.item() if torch.is_tensor(v) else v for k, v in summary_dict.items()
            }
        wandb.log(summary_dict, step=step, commit=commit)

    def close(self) -> None:
        wandb.finish()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    @classmethod
    def create(cls, wandb_config: WandbConfig) -> 'Writer':
        return cls(wandb_config)
