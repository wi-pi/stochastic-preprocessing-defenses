import argparse
import os
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm


@dataclass
class Config:
    exp_name: str
    module: str
    arguments: str
    log_file: str
    hparams: dict


class Experiment(object):
    """Utility for setting up experiments.
    """

    def __init__(self, config_file: Path, runs: list[str], log_dir: Path):
        """
        Create an experiment.
        :param config_file: Path to YAML config file.
        :param runs: Names of sub-experiments.
        :param log_dir: Directory to save / load logs.
        """
        with config_file.open() as fp:
            self.config = Config(**yaml.safe_load(fp))
        self.runs = runs or self.config.hparams.keys()
        self.log_dir = log_dir / self.config.exp_name

    def iter_hparams(self) -> Iterator[dict]:
        """Iterate over ALL combinations of arguments."""
        for hparam in map(self.config.hparams.get, self.runs):
            for vals in product(*hparam.values()):
                yield dict(zip(hparam.keys(), vals))

    def iter_commands(self, cmd_fmt: str) -> Iterator[str]:
        """For each set of arguments, yield its instantiated command."""
        for hp in self.iter_hparams():
            yield cmd_fmt.format(
                module=self.config.module,
                args=self.config.arguments.format_map(hp),
                output=self.log_file(hp)
            )

    def iter_results(self):
        """For each set of arguments, yield its output results."""
        for hp in tqdm(self.iter_hparams(), desc='Fetching results'):
            log_file = self.log_file(hp)
            if log_file.exists():
                with log_file.open() as fp:
                    results = re.findall(r'\((.+?)\): (\d+.\d+)\n', fp.read())  # format is like "(metric): xx.xx"
                yield hp | {k: float(v) for k, v in results}

    def log_file(self, hp: dict) -> Path:
        return self.log_dir / self.config.log_file.format_map(hp)


def _heat_map(df: pd.DataFrame, title: str, save: Path, aggfunc: str):
    plt.figure(constrained_layout=True)
    sns.set(font_scale=1.4)
    df = df.pivot_table(values='adaptive', index='eot', columns='step', aggfunc=aggfunc)
    ax = sns.heatmap(df, vmin=0, vmax=100, annot=True, fmt='.1f', linecolor='k', cmap='Blues', annot_kws={'size': 14})
    ax.invert_yaxis()
    plt.xlabel('PGD Steps')
    plt.ylabel('EOT Samples')
    plt.title(title)
    plt.savefig(save)


def plot(df: pd.DataFrame, metric: str, aggfunc: str, tag: str, root: Path):
    os.makedirs(root, exist_ok=True)
    for lr in [0.5, 1.0, 2.0, 4.0, 8.0]:
        title = f'{metric} (%) with LR = {lr:.1f}'
        save = root / f'{tag}_lr{lr:.1f}.pdf'
        _heat_map(df[df.lr == lr], title=title, save=save, aggfunc=aggfunc)

    _heat_map(df, f'{metric} (%) with LR = Best', root / f'{tag}_best.pdf', aggfunc=aggfunc)


def main(args):
    experiment = Experiment(config_file=args.file, runs=args.run, log_dir=args.log_dir)

    match args.action:

        case 'cmd':
            os.makedirs(experiment.log_dir, exist_ok=True)
            cmd_fmt = 'nohup python -u -m {module} {args} 2>&1 > {output}'
            for cmd in experiment.iter_commands(cmd_fmt):
                print(cmd)

        case 'plot' | 'view':
            df = pd.DataFrame(experiment.iter_results())

            if args.action == 'view':
                from IPython import embed
                embed(using=False)
                exit()

            # different metrics require different tags & aggregate functions
            match args.metric:
                case 'acc':
                    plot(df, 'Adversarial Accuracy', 'min', tag=experiment.config.exp_name, root=args.plot_dir)
                case 'asr':
                    plot(df, 'Attack Success Rate', 'max', tag=experiment.config.exp_name, root=args.plot_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('action', type=str, choices=['cmd', 'plot', 'view'])
    parser.add_argument('-r', '--run', type=str, nargs='+')
    parser.add_argument('--plot-dir', type=Path, default='./static/plots')
    parser.add_argument('--log-dir', type=Path, default='./static/logs')
    parser.add_argument('-m', '--metric', type=str, choices=['acc', 'asr'], default='acc')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
