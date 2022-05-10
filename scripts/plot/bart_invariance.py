import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def plot(tag: str, normal_asr: list, pretrain_asr: list):
    fig, ax1 = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax2 = ax1.twinx()

    # acc
    ax1.plot(x, normal_acc_benign, 'g--', lw=2.5)
    ax1.plot(x, pretrain_acc_benign, 'g-', lw=2.5)
    ax1.set_ylim(-2, 102)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylabel('Benign Accuracy (%)', color='g')

    # asr
    ax2.plot(x, normal_asr, 'k--', lw=2.5, label='before fine-tuning')
    ax2.plot(x, pretrain_asr, 'k-', lw=2.5, label='after fine-tuning')
    ax2.plot(x, normal_asr, 'r--', lw=2.5)
    ax2.plot(x, pretrain_asr, 'r-', lw=2.5)
    ax2.set_ylim(-2, 102)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Attack Success Rate (%)', color='r')

    plt.xticks(x)
    plt.xlabel('Number of Transformations')
    plt.legend(loc='lower left')

    # plt.show()
    plt.savefig(f'static/plots/invariance_bart_{tag}.pdf')


if __name__ == '__main__':
    mpl.rcParams['font.family'] = "Times"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['font.size'] = 20

    # load data
    df = pd.read_csv('static/bart_invariance.csv', index_col=0).dropna()
    df['epoch'] += 1

    """
    # >>> df = df.sort_values(['target', 'var'])
    # normal
    >>> untargeted = df[(df.target == -1) & (df.epoch == -1)]
    >>> targeted = df[(df.target == 9) & (df.epoch == -1)]
    # pretrain
    # python -m scripts.experiment experiments/imagenette_gaussian_autopgd.yml view
    >>> untargeted = df[(df.target == -1) & (df.epoch == 29)]
    >>> targeted = df[(df.target == 9) & (df.epoch == 29)]
    """
    x = [1, 2, 3, 4, 5]

    # untargeted
    # 100 - untargeted.benign.to_numpy()
    normal_acc_benign = [86.88, 86.75, 74.65, 74.14, 74.01]
    # untargeted.adaptive.to_numpy()
    normal_asr_adv_untargeted = [87.39, 59.91, 66.72, 79.55, 80.55]
    # 100 - untargeted.benign.to_numpy()
    pretrain_acc_benign = [94.65, 95.29, 92.48, 93.25, 92.48]
    # untargeted.adaptive.to_numpy()
    pretrain_asr_adv_untargeted = [98.79, 98.13, 96.97, 94.54, 96.01]

    # targeted
    # targeted.benign.to_numpy()
    normal_asr_benign = [8.28, 8.79, 9.3 , 9.55, 9.55]
    # targeted.adaptive.to_numpy()
    normal_asr_adv = [92.22, 55.31, 50.14, 84.51, 52.11]
    # targeted.benign.to_numpy()
    pretrain_asr_benign = [10.19,  9.94,  9.81, 10.19, 10.32]
    # targeted.adaptive.to_numpy()
    pretrain_asr_adv = [99.57, 98.02, 90.96, 96.17, 96.59]

    # plot_all()
    plot('untargeted', normal_asr_adv_untargeted, pretrain_asr_adv_untargeted)
    plot('targeted', normal_asr_adv, pretrain_asr_adv)
