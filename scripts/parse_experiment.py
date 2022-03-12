import itertools
import os.path

import yaml
import argparse


def gen_commands(data):
    cmd_fmt = 'nohup python -u -m {module} {args} 2>&1 > {output}'
    exp_name = data['exp_name']
    hparams = data['hparams']
    log_dir = data['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # enumerate over product of values
    for hp in itertools.product(*hparams.values()):
        # setup dict for current values
        hp = dict(zip(hparams.keys(), hp))

        # realize arguments
        args = data['arguments'].format_map(hp)

        # realize log file
        log_file = data['log_file'].format_map(hp)
        log_file = os.path.join(log_dir, f'{exp_name}_{log_file}')

        yield cmd_fmt.format(module=data['module'], args=args, output=log_file)


def main(args):
    with open(args.file, 'r') as f:
        data = yaml.safe_load(f)

    for cmd in gen_commands(data):
        print(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
