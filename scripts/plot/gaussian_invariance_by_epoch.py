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
    plt.figure()
    sns.lineplot(data=untargeted, x='epoch', y='benign', hue='var')
    plt.xlabel('Epoch')
    plt.ylabel('Benign Accuracy (%)')
    plt.ylim(0, 102)
    plt.title('Accuracy of Fine-tuning on Defense')
    plt.savefig(f'static/plots/invariance_gaussian_accuracy.pdf')

    # plot untargeted (by epoch)
    plt.figure()
    sns.lineplot(data=untargeted, x='epoch', y='adaptive', hue='var')
    plt.xlabel('Epoch')
    plt.ylabel('Attack Success Rate (%)')
    plt.ylim(0, 102)
    plt.title('Untargeted Attacks')
    plt.savefig(f'static/plots/invariance_gaussian_untargeted_by_epoch.pdf')

    # plot targeted (by epoch)
    plt.figure()
    sns.lineplot(data=targeted, x='epoch', y='adaptive', hue='var')
    plt.xlabel('Epoch')
    plt.ylabel('Attack Success Rate (%)')
    plt.ylim(0, 102)
    plt.title('Targeted Attacks')
    plt.savefig(f'static/plots/invariance_gaussian_targeted_by_epoch.pdf')

    # plot all (by var)
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twinx()

    # acc
    ax1.plot(x, untargeted[untargeted.epoch == 0].benign, 'g--', label='benign (low inv.)')
    ax1.plot(x, untargeted[untargeted.epoch == 9].benign, 'g-', label='benign (high inv.)')

    # asr
    ax2.plot(x, untargeted[untargeted.epoch == 0].adaptive, 'ro--', label='untargeted (low inv.)')
    ax2.plot(x, untargeted[untargeted.epoch == 9].adaptive, 'ro-', label='untargeted (high inv.)')
    ax2.plot(x, targeted[targeted.epoch == 0].adaptive, 'r^--', label='targeted (low inv.)')
    ax2.plot(x, targeted[targeted.epoch == 9].adaptive, 'r^-', label='targeted (high inv.)')

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.set_xlabel('Variance')
    ax1.set_ylabel('Benign Accuracy (%)', color='g')
    ax2.set_ylabel('Success Attack Rate (%)', color='r')

    plt.title(f'Attacks on Gaussian-Defended ImageNette')
    plt.xticks(x)
    fig.legend(loc='lower left', bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes)
    plt.savefig(f'static/plots/invariance_gaussian_all_by_var.pdf')


if __name__ == '__main__':
    main()
