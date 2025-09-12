from randomname import get_name

import wandb
from fax.config import Config


class WandbWriter:
    """Logging results to Weight and Biases.

    ### Args:
    `Config` cfg: full configuration of the experiment.
    """

    def __init__(self, cfg: Config) -> None:
        self.run = wandb.init(name=get_name(), config=cfg.to_dict(), reinit=True)
        return

    def log(self, data, step, commit=False) -> None:
        """Log results to wandb."""
        self.run.log(data, step=step, commit=commit)
        return

    def log_config(self, config: dict) -> None:
        """Log configuration to wandb."""
        self.run.config.update(config)
        return

    def update_summary(self, data: dict) -> None:
        """Log summary to wandb."""
        self.run.summary.update(data)
        return

    def finish(self) -> None:
        """Finish wandb run."""
        # push any remaining data to wandb by comitting an empty log
        self.run.log({}, commit=True)
        self.run.finish()
        return


class DummyLogger:
    """Dummy logger that does nothing."""

    def __init__(self, cfg: Config) -> None:
        _ = cfg
        return

    def log(self, data, step, commit=False) -> None:
        _, _, _ = data, step, commit
        return

    def log_config(self, config: dict) -> None:
        _ = config
        return

    def update_summary(self, data: dict) -> None:
        _ = data
        return

    def finish(self) -> None:
        return
