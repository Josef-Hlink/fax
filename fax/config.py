import tomllib
from pathlib import Path
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, Dict, Type

import attr

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
    n_gpus: int


@attr.s(auto_attribs=True, frozen=True)
class TrainingCFG:
    batch_size: int
    n_epochs: int
    n_samples: int
    n_eval_samples: int
    n_dataworkers: int


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


def parse_args(args: Namespace) -> CFG:
    cli_dict = vars(args)

    def build(section: str, cls: type):
        values = {}
        for field in attr.fields(cls):
            key_toml = field.name.replace('_', '-')  # default mapping
            cli_key = field.name
            if cli_key in cli_dict and cli_dict[cli_key] is not None:
                val = cli_dict[cli_key]
            else:
                val = DEFAULTS.get(section, {}).get(key_toml)
            # cast to field type
            if field.type is Path:
                val = Path(val).expanduser().resolve()
            elif field.type is not None:
                val = field.type(val)
            values[field.name] = val
        return cls(**values)

    return CFG(
        paths=build('PATHS', PathsCFG),
        base=build('BASE', BaseCFG),
        training=build('TRAINING', TrainingCFG),
        model=build('MODEL', ModelCFG),
        optim=build('OPTIM', OptimCFG),
    )


if __name__ == '__main__':
    parser = create_parser({'PATHS': 'iso exe logs', 'BASE': 'debug', 'OPTIM': 'lr wd'})
    args = parser.parse_args()
    config = parse_args(args)
    print(config)
