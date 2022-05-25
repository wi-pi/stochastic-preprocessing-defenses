import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data():
    # noisy (epoch starts from 1)
    df = pd.read_csv('static/gaussian_invariance_by_epoch.csv', index_col=0)
    df['epoch'] += 1
    var_list = df['var'].unique()

    # normal (epoch as 0)
    df_clean = pd.read_csv('static/gaussian_invariance_clean.csv', index_col=0)
    df_clean['epoch'] = 0
    df_clean = df_clean[df_clean['var'].isin(var_list)]

    # concat
    df = pd.concat([df, df_clean]).reset_index(drop=True)

    return df


def main():
    df = load_data()
    x = df['var'].unique()

    # parse data
    untargeted = df[df.target == -1]
    untargeted.benign = 100 - untargeted.benign
    targeted = df[df.target == 9]

    # plot accuracy (by epoch)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.lineplot(data=untargeted, x='epoch', y='benign', hue='var', ax=ax)
    plt.xlabel('Fine-tune Epochs')
    plt.ylabel('Benign Accuracy (%)')
    plt.ylim(0, 102)
    plt.savefig(f'static/plots/invariance_gaussian_accuracy.pdf')

    # plot untargeted (by epoch)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.lineplot(data=untargeted, x='epoch', y='adaptive', hue='var', ax=ax)
    plt.xlabel('# Epochs', fontsize=16)
    plt.ylabel('Attack Success Rate (%)', fontsize=16)
    plt.xticks(ticks=[0, 5, 10, 15, 20], fontsize=16)
    plt.yticks(fontsize=16)
    # plt.ylim(0, 102)
    plt.legend(title=r'Noise ($\sigma$)', loc='lower right')
    plt.savefig(f'static/plots/invariance_gaussian_untargeted_by_epoch.pdf')

    # plot targeted (by epoch)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.lineplot(data=targeted, x='epoch', y='adaptive', hue='var', ax=ax)
    plt.xlabel('# Epochs', fontsize=16)
    plt.ylabel('Attack Success Rate (%)', fontsize=16)
    plt.xticks(ticks=[0, 5, 10, 15, 20], fontsize=16)
    plt.yticks(fontsize=16)
    # plt.ylim(0, 102)
    plt.legend(title=r'Noise ($\sigma$)', loc='lower right')
    plt.savefig(f'static/plots/invariance_gaussian_targeted_by_epoch.pdf')


if __name__ == '__main__':
    mpl.rcParams['font.family'] = "Times"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['font.size'] = 14
    main()
