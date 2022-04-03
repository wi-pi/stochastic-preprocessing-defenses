# Non-Preprocessing Defenses

Implementations are adapted from https://github.com/wielandbrendel/adaptive_attacks_paper


## Special Requirements for Each Defense

### k-Winners-Take-All
```
foolbox==3.0.0
```

### Odds
```
conda create -n tf1 python=3.7
conda activate tf1
conda install cudatoolkit=10.0
conda install -c anaconda cudnn
pip install tensorflow-gpu==1.15 cleverhans==3.1.0 jupyter tqdm sklearn numpy
```
