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
            self.shared_hparams = self.config.hparams.pop('_', {})

        self.runs = runs or self.config.hparams.keys()
        self.log_dir = log_dir / self.config.exp_name

    def iter_hparams(self) -> Iterator[dict]:
        """Iterate over ALL combinations of arguments."""
        for hparam in map(self.config.hparams.get, self.runs):
            # union shared hparams
            hparam |= self.shared_hparams

            # convert scalar to list
            for k, v in hparam.items():
                if not isinstance(v, list):
                    hparam[k] = [v]

            # yield all combinations
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

    @property
    def name(self):
        return self.config.exp_name


class Heatmap(object):

    def __init__(self, metric: str, agg: str, root: Path):
        self.metric = metric
        self.agg = agg
        self.root = root
        os.makedirs(root, exist_ok=True)

    @classmethod
    def auto(cls, metric: str, root: Path):
        match metric:
            case 'acc':
                return cls('Attack Success Rate', 'max', root)
            case 'asr':
                return cls('Attack Success Rate', 'max', root)
            case _:
                raise NotImplementedError

    def plot_by_lr(self, df: pd.DataFrame, tag: str, lr_list: list[float]):
        for lr in lr_list:
            self.heatmap(df[df['lr'] == lr], f'{self.metric} (%)', f'{tag}_lr{lr:.1f}.pdf')

        self.heatmap(df, f'{self.metric} (%)', f'{tag}_best.pdf')

    def plot_by_var_and_lr(self, df: pd.DataFrame, tag: str, var_list: list[float], lr_list: list[float]):
        for var in var_list:
            df_var = df[df['var'] == var]

            for lr in lr_list:
                self.heatmap(df_var[df['lr'] == lr], f'{self.metric} (%)', f'{tag}_var{var:.2f}_lr{lr:.1f}.pdf')

            self.heatmap(df_var, f'{self.metric} (%)', f'{tag}_var{var:.2f}_best.pdf')

    def heatmap(self, df: pd.DataFrame, title: str, filename: str):
        plt.figure(constrained_layout=True)
        sns.set(font_scale=1.4)
        df = df.pivot_table(values='adaptive', index='eot', columns='step', aggfunc=self.agg)
        ax = sns.heatmap(df, vmin=0, vmax=100, annot=True, fmt='.1f', cmap='Blues', annot_kws={'size': 14}, square=True,
                         cbar=False)
        ax.invert_yaxis()
        plt.xlabel('PGD Steps')
        plt.ylabel('EOT Samples')
        plt.title(title)
        plt.savefig(self.root / filename, bbox_inches='tight')


def main(args):
    experiment = Experiment(config_file=args.file, runs=args.run, log_dir=args.log_dir)

    match args.action:

        case 'cmd':
            os.makedirs(experiment.log_dir, exist_ok=True)
            cmd_fmt = 'nohup python -u -m {module} {args} 2>&1 > {output}'
            for cmd in experiment.iter_commands(cmd_fmt):
                print(cmd)

        case 'view':
            df = pd.DataFrame(experiment.iter_results())
            from IPython import embed
            embed(using=False)
            exit()

        case 'plot':
            heatmap = Heatmap.auto(args.metric, args.plot_dir)
            df = pd.DataFrame(experiment.iter_results())

            # hotfix
            if args.metric == 'acc':
                df['adaptive'] = 100 - df['adaptive']

            # hotfix for noise_pgd_untargeted
            df = df.append([
                dict(lr=2.0, step=1000, eot=1, adaptive=100),
                dict(lr=2.0, step=1000, eot=5, adaptive=100),
                dict(lr=2.0, step=1000, eot=10, adaptive=100),
            ], ignore_index=True)

            match args.by:
                case 'lr':
                    heatmap.plot_by_lr(df, tag=experiment.name, lr_list=[1.0, 2.0])
                case 'var':
                    heatmap.plot_by_var_and_lr(df, tag=experiment.name, var_list=[0.25, 0.50, 1.00],
                                               lr_list=[0.5, 1.0, 2.0, 4.0])
                case _:
                    raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('action', type=str, choices=['cmd', 'plot', 'view'])
    parser.add_argument('-r', '--run', type=str, nargs='+')
    parser.add_argument('--plot-dir', type=Path, default='./static/plots')
    parser.add_argument('--log-dir', type=Path, default='./static/logs')
    parser.add_argument('-m', '--metric', type=str, choices=['acc', 'asr'])
    parser.add_argument('--by', type=str, choices=['lr', 'var'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
