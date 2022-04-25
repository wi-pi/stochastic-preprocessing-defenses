import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot(tag: str, normal_asr: list, pretrain_asr: list):
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twinx()

    # acc
    ax1.plot(x, normal_acc_benign, 'g--', label='low invariance')
    ax1.plot(x, pretrain_acc_benign, 'g-', label='high invariance')

    # asr
    ax2.plot(x, normal_asr, 'r--', label='low invariance')
    ax2.plot(x, pretrain_asr, 'r-', label='high invariance')

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.set_xlabel('Variance of Gaussian Noise')
    ax1.set_ylabel('Benign Accuracy (%)', color='g')
    ax2.set_ylabel('Success Attack Rate (%)', color='r')

    plt.title(f'{tag.title()} Attacks on Gaussian-Defended ImageNette')
    plt.xticks(x)
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')
    # plt.show()
    plt.savefig(f'static/plots/invariance_gaussian_{tag}.pdf')


def plot_all():
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twinx()

    # acc
    ax1.plot(x, normal_acc_benign, 'g--', label='benign (low inv.)')
    ax1.plot(x, pretrain_acc_benign, 'g-', label='benign (high inv.)')

    # asr
    ax2.plot(x, normal_asr_adv_untargeted, 'ro--', label='untargeted (low inv.)')
    ax2.plot(x, pretrain_asr_adv_untargeted, 'ro-', label='untargeted (high inv.)')
    ax2.plot(x, normal_asr_adv, 'r^--', label='targeted (low inv.)')
    ax2.plot(x, pretrain_asr_adv, 'r^-', label='targeted (high inv.)')

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.set_xlabel('Variance of Gaussian Noise')
    ax1.set_ylabel('Benign Accuracy (%)', color='g')
    ax2.set_ylabel('Success Attack Rate (%)', color='r')

    plt.title(f'Attacks on Gaussian-Defended ImageNette')
    plt.xticks(x)
    fig.legend(loc='lower left', bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes)
    plt.savefig(f'static/plots/invariance_gaussian.pdf')


if __name__ == '__main__':
    # mpl.rcParams['font.family'] = "Times New Roman"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['font.size'] = 12

    """
    >>> df = df.sort_values(['target', 'var'])
    # normal
    # python -m scripts.experiment experiments/imagenette_gaussian_autopgd_clean.yml view
    >>> untargeted = df[df.target == -1]
    >>> targeted = df[df.target == 9]
    # pretrain
    # python -m scripts.experiment experiments/imagenette_gaussian_autopgd.yml view
    >>> untargeted = df[(df.target == -1) & (df.epoch == 69)]
    >>> targeted = df[(df.target == 9) & (df.epoch == 69)]
    """
    x = np.arange(0.05, 0.55, 0.05)

    # untargeted
    # 100 - untargeted.benign.to_numpy()
    normal_acc_benign = [93.5, 81.53, 56.43, 37.32, 26.5, 21.4, 19.24, 18.34, 17.96, 14.9]
    # untargeted.adaptive.to_numpy()
    normal_asr_adv_untargeted = [99.05, 98.75, 94.58, 87.71, 81.73, 74.4, 64.24, 63.89, 62.41, 58.12]
    # 100 - untargeted.benign.to_numpy()
    pretrain_acc_benign = [95.92, 95.16, 94.52, 94.78, 93.38, 92.99, 92.99, 93.25, 91.21, 91.72]
    # untargeted.adaptive.to_numpy()
    pretrain_asr_adv_untargeted = [99.6, 99.06, 98.79, 98.12, 94.82, 92.33, 90.14, 85.11, 79.33, 75.69]

    # targeted
    # targeted.benign.to_numpy()
    normal_asr_benign = [9.55, 8.28, 5.99, 3.95, 1.78, 0.89, 0.51, 0., 0., 0.]
    # targeted.adaptive.to_numpy()
    normal_asr_adv = [99.86, 99.44, 91.87, 60.74, 22.44, 5.01, 1.15, 0.89, 0., 0.]
    # targeted.benign.to_numpy()
    pretrain_asr_benign = [9.81, 10.06, 9.94, 9.81, 10.06, 9.55, 10.06, 9.3, 9.68, 9.81]
    # targeted.adaptive.to_numpy()
    pretrain_asr_adv = [99.86, 99.72, 98.16, 95.62, 88.95, 80., 73.8, 64.04, 53.6, 49.44]

    plot_all()
    plot('untargeted', normal_asr_adv_untargeted, pretrain_asr_adv_untargeted)
    plot('targeted', normal_asr_adv, pretrain_asr_adv)
