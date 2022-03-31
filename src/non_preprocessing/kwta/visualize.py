import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load(lr: float, max_step: int, eot: int, sigma: float = 8.0):
    return pd.read_csv(args.log_dir / f'{sigma:.1f}_{lr:.2f}_{max_step}_{eot}.csv')


def load_by_lr(max_step: int, eot: int, lr_list: list[float]):
    for lr in lr_list:
        df = load(lr, max_step, eot)
        df['LR'] = lr
        yield df


def plot_no_eot():
    data = load_by_lr(max_step=40000, eot=1, lr_list=[0.1, 0.2, 0.5, 1.0])
    df = pd.concat(data, ignore_index=True)

    plt.figure(constrained_layout=True)
    ax = sns.lineplot(data=df, x='PGD Steps', y='Accuracy', hue='LR')
    ax.set(xscale="log")
    plt.savefig(args.plot_dir / args.output, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, default='static/logs/kwta')
    parser.add_argument('--plot-dir', type=Path, default='static/plots')
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    plot_no_eot()
