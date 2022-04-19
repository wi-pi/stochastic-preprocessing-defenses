import argparse
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger

from configs import DEFENSES, load_defense
from src.datasets import ImageNetteDataModule, CIFAR10DataModule
from src.models import CIFAR10ResNet, ImageNetteResNet


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--save', type=Path, default='./static/models/EOT-Attack')
    parser.add_argument('-v', '--version', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    # dataset
    parser.add_argument('--data', type=str, choices=['cifar10', 'imagenette'], required=True)
    parser.add_argument('--data-dir', type=Path, default='./static/datasets')
    # training
    parser.add_argument('-e', '--max-epochs', type=int, default=70)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--num-workers', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    # augment defenses
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+', default=[])
    parser.add_argument('-k', '--nb-defenses', type=int, default=1)
    parser.add_argument('-p', '--params', nargs='+', help='additional kwargs passed to defenses')
    args = parser.parse_args()

    # handle data dir
    match args.data:
        case 'imagenette':
            args.data_dir = args.data_dir / 'imagenette2-320'

    return args


def main(args):
    seed = pl.seed_everything()

    # Choose network and data module based on dataset
    match args.data:
        case 'cifar10':
            model_cls = CIFAR10ResNet
            dm_cls = CIFAR10DataModule
        case 'imagenette':
            model_cls = ImageNetteResNet
            dm_cls = ImageNetteDataModule
        case _:
            raise NotImplementedError(args.data)

    # Get defense
    defense = load_defense(defenses=args.defenses, nb_samples=args.nb_defenses, params=args.params)
    logger.debug(f'Defense: {defense}.')

    # Create pl modules
    model = model_cls(args.lr, args.wd, args.max_epochs)
    dm = dm_cls(args.data_dir, args.batch_size, args.num_workers, defense, seed)
    dm.prepare_data()

    trainer = pl.Trainer(
        logger=TensorBoardLogger(args.save, name=args.data, version=args.version),
        default_root_dir=args.save,
        max_epochs=args.max_epochs,
        gpus=[args.gpu],
        callbacks=[
            pl.callbacks.LearningRateMonitor('epoch'),
            pl.callbacks.TQDMProgressBar(),
            pl.callbacks.ModelCheckpoint(
                filename='epoch{epoch:02d}-acc{val_acc:.3f}', auto_insert_metric_name=False,
                monitor='val_acc', mode='max',
                # save_last=True, save_top_k=-1, every_n_epochs=10, save_weights_only=True,  # save every 10 epochs
                save_last=True, save_top_k=1, save_weights_only=True,  # save best + last
            ),
        ],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main(parse_args())
