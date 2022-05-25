import argparse
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load(lr: float, max_step: int, eot: int, sigma: float = 8.0):
    return pd.read_csv(args.log_dir / f'{sigma:.1f}_{lr:.2f}_{max_step}_{eot}.csv')


def load_by_lr(max_step: int, eot: int, lr_list: list[float]):
    for lr in lr_list:
        df = load(lr, max_step, eot)
        df['Gradient Queries'] = df['PGD Steps'] * eot
        df['LR'] = lr
        df['EOT'] = eot
        df['TAG'] = df.apply(lambda x: fr'$m$ = {int(x.EOT)}, $\alpha$ = {x.LR}', axis=1).astype('category')
        yield df


def plot_no_eot():
    data = load_by_lr(max_step=40000, eot=1, lr_list=[0.1, 0.2, 0.5, 1.0])
    df = pd.concat(data, ignore_index=True)

    plt.figure(constrained_layout=True)
    ax = sns.lineplot(data=df, x='PGD Steps', y='Accuracy', hue='LR')
    ax.set(xscale="log")
    plt.savefig(args.plot_dir / args.output, bbox_inches='tight')


def plot_all():
    concat = partial(pd.concat, ignore_index=True)
    df1 = concat(load_by_lr(max_step=40000, eot=1, lr_list=[0.1, 0.2, 0.5, 1.0]))
    df2 = concat(load_by_lr(max_step=20000, eot=100, lr_list=[0.5, 1.0, 2.0, 5.0]))
    df3 = concat(load_by_lr(max_step=100, eot=1000, lr_list=[0.5, 1.0, 2.0, 5.0]))
    df = concat([df1, df2, df3])
    df['Accuracy'] *= 100

    for col in ['PGD Steps', 'Gradient Queries']:
        plt.figure(figsize=(6, 5), constrained_layout=True)
        mpl.rcParams['font.family'] = "Times New Roman"
        mpl.rcParams['mathtext.fontset'] = "cm"
        mpl.rcParams['font.size'] = 15
        ax = sns.lineplot(data=df, x=col, y='Accuracy', hue='TAG')
        ax.set(xscale="log")
        plt.xlabel(f'# {col}')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='upper right')
        plt.savefig(args.plot_dir / f'{args.output}_{col.replace(" ", "_")}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, default='static/logs/kwta')
    parser.add_argument('--plot-dir', type=Path, default='static/plots')
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    plot_all()
