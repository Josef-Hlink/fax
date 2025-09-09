from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, Dict, Type

import attr


@attr.s(auto_attribs=True, frozen=True)
class Config:
    data_dir: str = ''
    debug: bool = False
    seed: int = 42
    n_gpus: int = 1
    batch_size: int = 64
    n_samples: int = 2**10
    n_eval_samples: int = 2**8
    n_epochs: int = 10
    seq_len: int = 256
    gamma: float = 0.999
    emb_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    lr: float = 3e-4
    b1: float = 0.9
    b2: float = 0.999
    wd: float = 1e-2
    grad_clip_norm: float = 1.0


@attr.s(auto_attribs=True, frozen=True)
class Help:
    data_dir = '[REQUIRED] Path to MDS data directory with train/val/test splits and stats.json'
    seed = 'Random seed for reproducibility'
    n_gpus = 'Number of GPUs to use for training [0 for CPU]'
    debug = 'More verbose logging'
    batch_size = 'Batch size for training'  # TODO: confirm
    n_samples = 'Number of training samples to use [-1 for all]'  # TODO: confirm
    n_eval_samples = 'Number of evaluation samples to use [-1 for all]'  # TODO: impl. -1
    n_epochs = 'Number of training epochs'
    seq_len = 'Sequence length (context size) for the model'
    gamma = 'Discount factor for sequence rewards calculation during preprocessing'
    emb_dim = 'Model embedding dimension'
    n_layers = 'Number of transformer layers'
    n_heads = 'Number of attention heads'
    dropout = 'Dropout rate for the model'
    lr = 'Learning rate for the optimizer'
    b1 = 'Beta1 for AdamW optimizer'
    b2 = 'Beta2 for AdamW optimizer'
    wd = 'Weight decay for AdamW optimizer'
    grad_clip_norm = 'Gradient clipping norm value'


def create_parser(cls: Type[Any], parser: ArgumentParser) -> ArgumentParser:
    help_messages = Help()
    for field in attr.fields(cls):
        arg_name = f'--{field.name.replace("_", "-")}'
        if field.type == bool:
            parser.add_argument(
                arg_name, action='store_true', help=getattr(help_messages, field.name, '')
            )
            continue
        parser.add_argument(
            arg_name,
            type=field.type,
            default=field.default if field.default is not attr.NOTHING else None,
            help=getattr(help_messages, field.name, ''),
            required=field.type == str and field.default == '',  # only for data_dir
        )

    return parser


def parse_args(cls: Type[Any], args: Namespace) -> Config:
    kwargs: Dict[str, Any] = {}
    for field in attr.fields(cls):
        value = getattr(args, field.name, field.default)
        kwargs[field.name] = value
    return cls(**kwargs)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='FAX Training Configuration', formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser = create_parser(Config, parser)
    config = parse_args(Config, parser.parse_args())

    for k, v in config.__dict__.items():
        print(f'{k}: {v}')
