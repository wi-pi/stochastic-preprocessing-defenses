import os
from operator import itemgetter
from typing import Optional

import gpustat
import numpy as np
from loguru import logger


def set(gpu: int):
    current_gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if current_gpu:
        logger.warning(f'GPU already set to {current_gpu}.')
        return

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    logger.debug(f'Setting GPU to {gpu}.')


def setgpu(gpu: Optional[int] = None, gb: float = 1.0):
    if gpu is not None:
        return set(gpu)

    fetch = itemgetter('index', 'memory.used', 'memory.total')
    data = [fetch(gpu) for gpu in gpustat.GPUStatCollection.new_query()]
    data = np.array(data)

    # take the best
    usage = data[:, 1] / data[:, 2]
    gpu = data[usage.argsort()][0]

    # check valid
    if gpu[2] - gpu[1] >= gb * 1024:
        return set(gpu[0])

    raise RuntimeError(f'Cannot find GPU with sufficient memory {gb} GB.')


if __name__ == '__main__':
    setgpu()
