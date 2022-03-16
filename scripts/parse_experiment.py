from dataclasses import dataclass
import itertools
import os

import pandas as pd
import yaml
import argparse

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class Config:
    exp_name: str
    module: str
    arguments: str
    log_dir: str
    log_file: str
    hparams: dict


class ExperimentParser(object):

    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.config = Config(**yaml.safe_load(f))

    def iter_hparams(self):
        for hp in itertools.product(*self.config.hparams.values()):
            yield dict(zip(self.config.hparams.keys(), hp))

    def iter_commands(self, cmd_fmt: str):
        os.makedirs(self.config.log_dir, exist_ok=True)
        for hp in self.iter_hparams():
            args = self.config.arguments.format_map(hp)
            log_file = self.log_file(hp)
            yield cmd_fmt.format(module=self.config.module, args=args, output=log_file)

    def iter_results(self, cmd_fmt: str):
        for hp in tqdm(self.iter_hparams()):
            log_file = self.log_file(hp)
            if os.path.exists(log_file):
                row_keys = ['benign', 'non-adaptive', 'adaptive']
                row_vals = [float(line.strip()) for line in os.popen(cmd_fmt.format(log_file=log_file))]
                if len(row_keys) == len(row_vals):
                    yield hp | dict(zip(row_keys, row_vals))

    def log_file(self, hp: dict):
        log_file = self.config.log_file.format_map(hp)
        return os.path.join(self.config.log_dir, f'{self.config.exp_name}_{log_file}')


def heat_map(df: pd.DataFrame, title: str, save: str):
    plt.figure(constrained_layout=True)
    sns.set(font_scale=1.4)
    df = df.pivot_table(values='adaptive', index='eot', columns='step', aggfunc='max')
    ax = sns.heatmap(df, vmin=0, vmax=100, annot=True, fmt='.1f', cmap='Blues', annot_kws={'size': 14})
    ax.invert_yaxis()
    plt.xlabel('PGD Steps')
    plt.ylabel('EOT Samples')
    plt.title(title)
    plt.savefig(save)


def plot(df: pd.DataFrame, metric: str, tag: str, root: str):
    os.makedirs(root, exist_ok=True)
    for lr in [0.5, 1.0, 2.0, 4.0, 8.0]:
        title = f'Attack Success Rate (%) with LR = {lr:.1f}'
        save = os.path.join(root, f'{tag}_lr{lr:.1f}.pdf')
        heat_map(df[df.lr == lr], title=title, save=save)

    heat_map(df, f'{metric} (%) with LR = Best', os.path.join(root, f'{tag}_best.pdf'))


def main(args):
    parser = ExperimentParser(args.file)
    match args.action:
        case 'cmd':
            cmd_fmt = 'nohup python -u -m {module} {args} 2>&1 > {output}'
            for cmd in parser.iter_commands(cmd_fmt):
                print(cmd)
        case 'plot':
            cmd_fmt = "grep Rate {log_file} | awk '{{print $NF}}'"
            df = pd.DataFrame(list(parser.iter_results(cmd_fmt)))
            plot(df, metric='Attack Success Rate', tag=parser.config.exp_name, root='./static/plots')
            # plot(df, metric='Adversarial Accuracy', tag=parser.config.exp_name, root='./static/plots')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('action', type=str, choices=['cmd', 'plot'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
