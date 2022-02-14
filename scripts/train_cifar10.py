import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.datasets.cifar10 import CIFAR10DataModule
from src.models.cifar10 import CIFAR10ResNet


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--save', type=str, default='./static/logs')
    parser.add_argument('--data-dir', type=str, default='./static/datasets')
    parser.add_argument('-v', '--version', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    # training
    parser.add_argument('-e', '--max-epochs', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--num-workers', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    # augment defenses
    parser.add_argument('-d', '--defenses', type=float, nargs='+')
    args = parser.parse_args()
    return args


def main(args):
    seed = pl.seed_everything()

    model = CIFAR10ResNet(args.lr, args.wd, args.max_epochs)
    dm = CIFAR10DataModule(args.data_dir, args.batch_size, args.num_workers, args.defenses, seed)
    dm.prepare_data()

    trainer = pl.Trainer(
        logger=TensorBoardLogger(args.save, name='', version=args.version),
        default_root_dir=args.save,
        max_epochs=args.max_epochs,
        gpus=[args.gpu],
        callbacks=[
            pl.callbacks.LearningRateMonitor('epoch'),
            pl.callbacks.TQDMProgressBar(),
            pl.callbacks.ModelCheckpoint(
                filename='epoch{epoch:02d}-acc{val_acc:.3f}', auto_insert_metric_name=False,
                monitor='val_acc', mode='max', save_top_k=1, save_weights_only=True,
            ),
        ],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main(parse_args())
