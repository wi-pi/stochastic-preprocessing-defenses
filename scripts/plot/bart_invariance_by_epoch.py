import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # load data
    df = pd.read_csv('static/bart_invariance.csv', index_col=0).dropna()
    df['epoch'] += 1
    df = df[df['v'] <= 4]

    # parse data
    untargeted = df[df.target == -1]
    untargeted.benign = 100 - untargeted.benign
    targeted = df[df.target == 9]

    # plot accuracy (by epoch)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.lineplot(data=untargeted, x='epoch', y='benign', hue='v', ax=ax)
    plt.xlabel('Fine-tune Epochs')
    plt.ylabel('Benign Accuracy (%)')
    plt.legend(title='# Transformations', loc='lower right')
    # plt.show()
    plt.savefig(f'static/plots/invariance_bart_accuracy.pdf')

    # plot untargeted (by epoch)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.lineplot(data=untargeted, x='epoch', y='adaptive', hue='v', ax=ax)
    plt.xlabel('Fine-tune Epochs')
    plt.ylabel('Attack Success Rate (%)')
    plt.legend(title='# Transformations', loc='lower right')
    # plt.show()
    plt.savefig(f'static/plots/invariance_bart_untargeted_by_epoch.pdf')

    # plot targeted (by epoch)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    sns.lineplot(data=targeted, x='epoch', y='adaptive', hue='v', ax=ax)
    plt.xlabel('Fine-tune Epochs')
    plt.ylabel('Attack Success Rate (%)')
    plt.legend(title='# Transformations', loc='lower right')
    # plt.show()
    plt.savefig(f'static/plots/invariance_bart_targeted_by_epoch.pdf')


if __name__ == '__main__':
    mpl.rcParams['font.family'] = "Times"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['font.size'] = 13
    main()
