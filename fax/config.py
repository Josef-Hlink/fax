import sys
import tomllib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Dict, List, Type, get_args, get_origin

import attr
from loguru import logger

# project root; this is <proj>/fax/config.py
_PROJ = Path(__file__).parent.parent.resolve()
with open(_PROJ / 'defaults.toml', 'rb') as f:
    DEFAULTS = tomllib.load(f)
with open(_PROJ / 'help.toml', 'rb') as f:
    HELP = tomllib.load(f)


@attr.s(auto_attribs=True, frozen=True)
class PathsCFG:
    iso: Path
    exe: Path
    zips: List[Path]
    slp: Path
    sql: Path
    mds: Path
    runs: Path
    logs: Path
    replays: Path
    dolphin_home: Path


@attr.s(auto_attribs=True, frozen=True)
class BaseCFG:
    seed: int
    debug: bool
    wandb: bool
    n_gpus: int


@attr.s(auto_attribs=True, frozen=True)
class TrainingCFG:
    batch_size: int
    n_epochs: int
    n_samples: int
    n_val_samples: int
    n_dataworkers: int

    def __attrs_post_init__(self):
        """Validate that batch_size, n_samples, and n_val_samples are powers of 2."""
        if self.batch_size & (self.batch_size - 1) != 0:
            raise ValueError(f'batch_size must be a power of 2, got {self.batch_size}')
        if self.n_samples & (self.n_samples - 1) != 0:
            raise ValueError(f'n_samples must be a power of 2, got {self.n_samples}')
        if self.n_val_samples & (self.n_val_samples - 1) != 0:
            raise ValueError(f'n_val_samples must be a power of 2, got {self.n_val_samples}')


@attr.s(auto_attribs=True, frozen=True)
class ModelCFG:
    n_layers: int
    n_heads: int
    seq_len: int
    emb_dim: int
    dropout: float
    gamma: float


@attr.s(auto_attribs=True, frozen=True)
class OptimCFG:
    lr: float
    wd: float
    b1: float
    b2: float


@attr.s(auto_attribs=True, frozen=True)
class CFG:
    paths: PathsCFG
    base: BaseCFG
    training: TrainingCFG
    model: ModelCFG
    optim: OptimCFG

    def to_dict(self) -> dict:
        """Convert the CFG dataclass to a dictionary."""
        return {
            'paths': attr.asdict(self.paths),
            'base': attr.asdict(self.base),
            'training': attr.asdict(self.training),
            'model': attr.asdict(self.model),
            'optim': attr.asdict(self.optim),
        }


def create_parser(argnames: Dict[str, str]) -> ArgumentParser:
    """Create and add arguments to an ArgumentParser from a dictionary of argument names and help strings.

    Args:
        argnames: A dictionary where keys are sections (e.g., 'paths', 'base')
            and values are space-separated argument names (e.g., 'data-dir batch-size').
    Returns: The updated ArgumentParser.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    for section, names in argnames.items():
        for name in names.split():
            arg_name = f'--{name}'
            help_msg = HELP.get(section, {}).get(name, '')
            default = DEFAULTS.get(section, {}).get(name, None)

            if section == 'PATHS':
                if isinstance(default, list):
                    parser.add_argument(
                        arg_name,
                        type=Path,
                        nargs='+',
                        default=[Path(d).expanduser().resolve() for d in default],
                        help=help_msg,
                    )
                    continue
                default = Path(default).expanduser().resolve()
            if isinstance(default, bool):
                parser.add_argument(arg_name, action='store_true', help=help_msg)
            else:
                parser.add_argument(
                    arg_name,
                    type=type(default) if default is not None else str,
                    default=default,
                    help=help_msg,
                )
    return parser


def parse_args(args: Namespace, caller: str) -> CFG:
    """Parse command-line arguments and merge them with defaults from the TOML file.
    Args:
        args: Parsed command-line arguments.
        caller: Name of the calling script (just pass __file__).
    Returns: The complete configuration as a CFG object.
    """
    cli_dict = vars(args)

    def build(section: str, cls: Type):
        values = {}
        for field in attr.fields(cls):
            toml_key = field.name.replace('_', '-')  # default mapping
            cli_key = field.name
            if cli_key in cli_dict and cli_dict[cli_key] is not None:
                val = cli_dict[cli_key]
            else:
                val = DEFAULTS.get(section, {}).get(toml_key)
            # cast to Path object
            if field.type is Path:
                val = Path(val).expanduser().resolve()
            # for zips: List[Path] with globbing
            elif get_origin(field.type) is list and get_args(field.type)[0] == Path:
                # normalize: allow str or list[str|Path]
                raw_vals = val if isinstance(val, list) else [val]
                _val = []
                for v in raw_vals:
                    v_path = Path(v).expanduser().resolve()  # works whether str or Path
                    _val.extend([Path(p).expanduser().resolve() for p in glob(v_path.as_posix())])
                val = _val
            # int, float, bool (and possibly str but we don't use those)
            elif field.type is not None:
                val = field.type(val)
            values[field.name] = val
        return cls(**values)

    cfg = CFG(
        paths=build('PATHS', PathsCFG),
        base=build('BASE', BaseCFG),
        training=build('TRAINING', TrainingCFG),
        model=build('MODEL', ModelCFG),
        optim=build('OPTIM', OptimCFG),
    )
    setup_logger(cfg.paths.logs / Path(caller).stem, debug=cfg.base.debug)
    for section_name, section in cfg.__dict__.items():
        for key, value in section.__dict__.items():
            logger.debug(f'Config {section_name}.{key} = {value}')
    return cfg


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


if __name__ == '__main__':
    exposed_args = {
        'PATHS': 'iso exe zips slp sql mds runs logs replays dolphin-home',
        'BASE': 'seed debug wandb n-gpus',
        'TRAINING': 'batch-size n-epochs n-samples n-val-samples n-dataworkers',
        'MODEL': 'n-layers n-heads seq-len emb-dim dropout gamma',
        'OPTIM': 'lr wd b1 b2',
    }
    parser = create_parser(exposed_args)
    args = parser.parse_args()
    config = parse_args(args, caller=__file__)
    for section_name, section in exposed_args.items():
        print(f'[{section_name}]')
        sec = getattr(config, section_name.lower())
        for key in section.split():
            print(f'{key}: {getattr(sec, key.replace("-", "_"))}')
        print()
