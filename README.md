[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![figure](docs/imagenet_collage.png)

# Data Augmentation with Bilevel Optimization [[Paper]](https://arxiv.org/pdf/2006.14699.pdf)
The goal is to automatically learn an efficient data augmentation regime for image
classification.



## Table of Contents

- [Overview](#usage)
- [Experiments](#experiments)
- [Citation](#citation)

## Overview

<b>What's new:</b>



Image 3 here

## Why it matters

Image 4 here

## Experiments

### Install requirements
`pip install -r requirements.txt` 
This command installs the Haven library which helps in managing the experiments.

### 2.1 MNIST
`python trainval.py -e mnist -sb ../results -d ../data -r 1`

where `-e` is the experiment group, `-sb` is the result directory, and `-d` is the dataset directory.

#### 2.2 Cifar100 experiment

`python trainval.py -e cifar100 -sb ../results -d ../data -r 1`

### 3. Results
#### 3.1 Launch Jupyter by running the following on terminal,

```
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook
```

#### 3.2 On a Jupyter cell, run the following script,
```python
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu

# path to where the experiments got saved
savedir_base = '../results'

# filter exps
filterby_list = [{'dataset':'cifar100', 'opt':{'c':0.5}}, 
                 {'dataset':'cifar100', 'opt':{'name':'adam'}}]
                 
# get experiments
rm = hr.ResultManager(savedir_base=savedir_base, 
                      filterby_list=filterby_list, 
                      verbose=0)
                      
# dashboard variables
legend_list = ['opt.name']
title_list = ['dataset', 'model']
y_metrics = ['train_loss', 'val_acc']

# launch dashboard
hj.get_dashboard(rm, vars(), wide_display=True)
```


![alt text](neurips2019/cifar100.jpg)


#### Citation

```
@article{mounsaveng2020learning,
  title={Learning Data Augmentation with Online Bilevel Optimization for Image Classification},
  author={Mounsaveng, Saypraseuth and Laradji, Issam and Ayed, Ismail Ben and Vazquez, David and Pedersoli, Marco},
  journal={arXiv preprint arXiv:2006.14699},
  year={2020}
}
```
