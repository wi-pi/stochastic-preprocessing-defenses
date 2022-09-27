# On the Limitations of Stochastic Pre-processing Defenses

This repository is the official implementation of *On the Limitations of Stochastic Pre-processing Defenses*.

## Requirements

### Environments

To install requirements:

```shell
conda create -n your_env_name python=3.10
conda activate your_env_name
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

### Datasets

To prepare ImageNet:
* Download the validation set from https://www.image-net.org
* Extract to `./static/datasets/` so that the file structure becomes:
    ```
    src/
    static/datasets/imagenet/val/
      n01440764/*.JPEG
      n01775062/*.JPEG
      ...
      n07579787/*.JPEG
    ```

To prepare ImageNette:
* Download the full dataset ("320 px") from https://github.com/fastai/imagenette
* Extract to `./static/datasets/` so that the file structure becomes:
    ```
    src/
    static/datasets/imagenette2-320/
      train/
        n01440764/*.JPEG
        n02102040/*.JPEG
        ...
        n03888257/*.JPEG
      val/
        n01440764/*.JPEG
        n02102040/*.JPEG
        ...
        n03888257/*.JPEG
    ```

### Pre-trained Models

To prepare models fine-tuned on ImageNet and Gaussian noise
* Download models from https://github.com/locuslab/smoothing
* Extract to `./static/models/` so that the file structure becomes:
    ```
    src/
    static/models/smoothing-models/imagenet/resnet50/
      noise_0.25/checkpoint.pth.tar
      noise_0.50/checkpoint.pth.tar
    ```

To prepare models fine-tuned on ImageNette
* Download models from the anonymous Google Drive https://drive.google.com/drive/folders/15qnyQ8Q8t271vbcfvpm_CpiV1xavLaGP
* Extract to `./static/models/`

We provide three models pre-trained on ImageNette:

| Filename                                  | Defenses                          | Top-1 Accuracy (%) |
|-------------------------------------------|-----------------------------------|:------------------:|
| `ResNet34-ImageNette-Clean.ckpt`          | N/A                               |       96.31%       |
| `ResNet34-ImageNette-NoiseInjection.ckpt` | Noise Injection                   |       94.65%       |
| `ResNet34-ImageNette-Gaussian0.50.ckpt`   | Randomized Smoothing (sigma 0.50) |       92.36%       |

## Usage

### Evaluate Defenses on ImageNet

To evaluate Random Rotation with targeted PGD-50 (eps 8/255, lr 1/255), EOT-1, and Vote 20:

```shell
python -m scripts.test_imagenet \
    --load r50 --mode vote --repeat 20 \
    --data-dir static/datasets/imagenet --data-skip 50 --batch 100 \
    --attack pgd --norm inf --eps 8 --lr 1 --step 50 --eot 1 --target 9 --random-init 1 \
    --defense Rotation --params degree=90
```

To evaluate Randomized Smoothing (sigma 0.25) with targeted PGD-50 (eps 8/255, lr 1/255), EOT-1, and Vote 500:

```shell
python -m scripts.test_imagenet \
    --load r50-s0.25 --mode vote --repeat 500 \
    --data-dir static/datasets/imagenet --data-skip 50 --batch 100 \
    --attack pgd --norm inf --eps 8 --lr 1 --step 50 --eot 1 --target 9 --random-init 1 \
    --defense GaussianNoisePyTorchNoClip --params variance=0.25
```

### Fine-tune Models on ImageNette Processed by Defenses

To fine-tune the model on data processed by Noise Injection:

```shell
python -m scripts.train \
    --data imagenette --data-dir static/datasets --save static/models --version test \
    --max-epochs 30 --batch-size 256 --num-workers 16 \
    --lr 1e-3 --wd 1e-2 --load clean \
    --defenses NoiseInjectionPyTorch
```

To fine-tune the model on data processed by Gaussian noise (sigma 0.50):

```shell
python -m scripts.train \
    --data imagenette --data-dir static/datasets --save static/models --version test \
    --max-epochs 30 --batch-size 256 --num-workers 16 \
    --lr 1e-3 --wd 1e-2 --load clean \
    --defenses GaussianNoisePyTorch -p variance=0.50
```

### Evaluate Defenses on ImageNette

To evaluate Noise Injection on the model *before* fine-tuning:

```shell
python -m scripts.test_imagenette \
    --load path/to/your/not/fine-tuned/model.ckpt \
    --mode vote --repeat 500 \
    --data-dir static/datasets/imagenette2-320 --data-skip 5 --batch 100 \
    --attack pgd --norm inf --eps 8 --lr 1 --step 50 --eot 1 --target 9 --random-init 1 \
    --defenses NoiseInjectionPyTorch
```

To evaluate Noise Injection on the model *after* fine-tuning:

```shell
python -m scripts.test_imagenette \
    --load path/to/your/fine-tuned/model.ckpt \
    --mode vote --repeat 500 \
    --data-dir static/datasets/imagenette2-320 --data-skip 5 --batch 100 \
    --attack pgd --norm inf --eps 8 --lr 1 --step 50 --eot 1 --target 9 --random-init 1 \
    --defenses NoiseInjectionPyTorch
```

### Miscellaneous

To save the ImageNet image (ID 0) processed by Random Rotation (10 samples):

```shell
python -m scripts.visualize_defense \
    --dataset imagenet --data-dir static/datasets/imagenet \
    --id 0 -n 10 \
    --defense Rotation --params degree 90 \
    --save path/to/outputs --tag rotation90
```

## Citation

If you find this work useful in your research, please cite our paper with the following BibTeX:

```bib
@inproceedings{gao2022limitations,
  author    = {Yue Gao and Ilia Shumailov and Kassem Fawaz and Nicolas Papernot},
  title     = {On the Limitations of Stochastic Pre-processing Defenses},
  booktitle = {Thirty-Sixth Conference on Neural Information Processing Systems},
  year      = {2022},
  url       = {https://openreview.net/forum?id=P_eBjUlzlV}
}
```
