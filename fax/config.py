from typing import Type, Any, Tuple, Dict
from argparse import ArgumentParser, Namespace

import attr


@attr.s(auto_attribs=True, frozen=True)
class BaseConfig:
    seed: int = 42
    n_gpus: int = 1  # set to 0 for CPU training
    debug: bool = False


@attr.s(auto_attribs=True, frozen=True)
class DataConfig:
    """Separated for easy passing around."""

    dir: str = ''
    batch_size: int = 64
    n_samples: int = 2**10
    max_samples: int = -1  # -1 means no limit
    n_eval_samples: int = 2**8
    n_epochs: int = 4
    gamma: float = 0.999  # for discounted returns in preprocessing


@attr.s(auto_attribs=True, frozen=True)
class ModelConfig:
    block_size: int = 256  # context length
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 8
    dropout: float = 0.1


@attr.s(auto_attribs=True, frozen=True)
class TrainConfig:
    """Inherits from BaseConfig."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    wd: float = 1e-2  # weight decay
    grad_clip_norm: float = 1.0

    @property
    def betas(self) -> Tuple[float, float]:
        return (self.beta1, self.beta2)


def check_required(field: Any) -> bool:
    return field.default is attr.NOTHING or (field.type == str and field.default == '')


def create_parser(cls: Type[Any], parser: ArgumentParser, prefix: str = '') -> ArgumentParser:
    for field in attr.fields(cls):
        arg_name = f'--{prefix}{field.name.replace("_", "-")}'
        if attr.has(field.type):
            # nested dataclass -> recurse
            parser = create_parser(field.type, parser, prefix=f'{field.name}.')
        else:
            if field.type == bool:
                parser.add_argument(
                    arg_name, action='store_true', help=field.metadata.get('help', '')
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=field.type,
                    default=field.default if field.default is not attr.NOTHING else None,
                    help=field.metadata.get('help', ''),
                    required=check_required(field),
                )

    return parser


def parse_args(cls: Type[Any], args: Namespace, prefix: str = '') -> Any:
    kwargs: Dict[str, Any] = {}
    for field in attr.fields(cls):
        arg_name = f'{prefix}{field.name.replace("_", "-")}'
        if attr.has(field.type):
            # nested dataclass -> recurse
            value = parse_args(field.type, args, prefix=f'{field.name}.')
        else:
            value = getattr(args, arg_name.replace('-', '_'), field.default)
            if value is None and field.default is attr.NOTHING:
                raise ValueError(f'Missing required argument: {arg_name}')
        kwargs[field.name] = value
    return cls(**kwargs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = create_parser(BaseConfig, parser)
    parser = create_parser(TrainConfig, parser, prefix='train.')
    args = parser.parse_args()

    base_config = parse_args(BaseConfig, args)
    train_config = parse_args(TrainConfig, args, prefix='train.')

    print(base_config)
    print(train_config)
    print(train_config.data)
    print(train_config.model)
