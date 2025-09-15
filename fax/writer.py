# -*- coding: utf-8 -*-

from typing import Dict, Optional

import wandb

from fax.config import CFG


class WandbWriter:
    """Logging results to Weight and Biases.

    ### Args:
    `Config` cfg: full configuration of the experiment.
    """

    def __init__(self, cfg: CFG, run_name: str) -> None:
        self.run = wandb.init(name=run_name, config=cfg.to_dict())
        return

    def log(self, data: Dict, step: Optional[int], commit=False) -> None:
        """Log results to wandb."""
        self.run.log(data, step=step, commit=commit)
        return

    def update_summary(self, data: Dict) -> None:
        """Log summary to wandb."""
        self.run.summary.update(data)
        return

    def finish(self) -> None:
        """Finish wandb run."""
        # push any remaining data to wandb by comitting an empty log
        self.run.log({}, commit=True)
        self.run.finish()
        return


class DummyWriter:
    """Dummy writer that does nothing."""

    def __init__(self, cfg: CFG, run_name: str) -> None:
        _ = cfg, run_name
        return

    def log(self, data: Dict, step: Optional[int], commit=False) -> None:
        _ = data, step, commit
        return

    def update_summary(self, data: Dict) -> None:
        _ = data
        return

    def finish(self) -> None:
        return
