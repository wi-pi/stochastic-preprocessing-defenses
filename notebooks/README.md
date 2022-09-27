# Experiments for Section 4

This directory provides notebooks for experiments in Section 4.

## Preview

We provide previews for these notebooks as HTML files, which can be opened in browsers.


## Evaluate Random Rotation

We use the official evaluation code from https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/notebooks.

To set up environment:

* Setup environment as in our main evaluation code.
* Run `pip install tensorflow_addons`.
* Run `pip install git+https://github.com/nottombrown/imagenet_stubs`.

To run notebooks:

```shell
jupyter notebook --no-browser
```


## Evaluate Random Cropping and Rescaling

We use the official evaluation code from https://github.com/anishathalye/obfuscated-gradients.

*Since their code uses the outdated TensorFlow v1, the setup below may not work in all environments.*

To set up environment:

```shell
conda create -n your_env_name python=3.7
conda activate your_env_name
conda install cudatoolkit=10.0
conda install -c anaconda cudnn
pip install tensorflow-gpu==1.15 cleverhans==3.1.0 robustml==0.0.3 jupyter tqdm sklearn numpy
````

To set up models:

* Download weight file http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
* Extract to `data/inception_v3.ckpt` in each folder.

To run notebooks:

```shell
jupyter notebook --no-browser
```
