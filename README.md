# bilvlda

### TODO

- Fix outer loop
- Reproduce results for CIFAR10 with 95% test accuracy



### EXPERIMENTS

|  Status | Command | Score  | Description | 
| --- | --- | --- | --- |
| debug | `python trainval.py -e cifar_saypra -sb <savedir_base> -r 1` | TBA (epoch 100) | Train small-affine and Clf on Cifar10|

### Timeline

- Week 1: setup baselines on cifar and mnist



##  Configurations

The following should give 94.5% accuracy in CIFAR10
```python
{
            "name": "saypra",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/trainval.py",
            "console": "integratedTerminal",
            "args":[
                "-e", "cifar_saypra",
                "-sb", "/mnt/datasets/public/issam/prototypes/bilvada/non_borgy/",
                "-r", "1"
        ],
        },
```

Bilevel optimization for data augmentation

Todo:
MixTrainVal \\
BACH dataset \\
Check if everything needed in case of restart is saved correctly (seed, train/val indexes) \\
Save reference images \\
Vizualisation
