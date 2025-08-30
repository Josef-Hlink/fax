# fax/dataloader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Any, Dict
from streaming import StreamingDataset

from fax.config import TrainConfig, create_parser, parse_args
from fax.stats import load_dataset_stats


class MDSDataLoader:
    def __init__(self, data_dir: Path, split: str = 'train'):
        self.data_dir = data_dir
        self.split = split
        self.stats = load_dataset_stats(data_dir)
        print(f'[MDSDataLoader] init: dir={data_dir!r} split={split}')
        print(f'[MDSDataLoader] init: stats keys={list(self.stats.keys())[:3]}')
        # NOTE: maybe investigate batch size
        self.ds = StreamingDataset(
            local=data_dir.as_posix(), remote=None, split=split, shuffle=False, batch_size=1024
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        print('[MDSDataLoader] iter: starting stream')
        for sample in self.ds:
            yield sample


if __name__ == '__main__':
    # minimal debug entrypoint: parse your TrainConfig and preview a few samples
    import argparse

    parser = argparse.ArgumentParser()
    parser = create_parser(TrainConfig, parser, prefix='train.')
    args = parser.parse_args()
    train_cfg = parse_args(TrainConfig, args, prefix='train.')

    print(f'[main] train.data.dir = {train_cfg.data.dir!r}')
    loader = MDSDataLoader(Path(train_cfg.data.dir), split='train')

    for i, s in enumerate(loader):
        keys = list(s.keys())
        head = keys[:3]
        print(f'[main] sample #{i + 1}: keys={head}{" ..." if len(keys) > 10 else ""}')
        shapes = {k: getattr(s[k], 'shape', None) for k in head}
        print(f'[main] shapes: {shapes}')
        if i >= 2:
            break
