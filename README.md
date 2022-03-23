# EOT-Attack
> Evaluate stochastic defenses and EOT attacks.

## Python Environment
```shell
conda create -n eot python=3.10
conda activate eot

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

## Evaluations

### Single Run

**Evaluate CIFAR10**

`python -m scripts.test_cifar10 --help`

**Evaluate ImageNet**

`python -m scripts.test_imagenet --help`

**Visualize Defense**

`python -m scripts.visualize_defense --help`

### Batch Run

**Execute commands**

`python -m scripts.experiment CONFIG.yml cmd | simple_gpu_scheduler --gpus 0,1,2,3`

**View data**

`python -m scripts.experiment CONFIG.yml view`

**Plot figures**

`python -m scripts.experiment CONFIG.yml plot`
