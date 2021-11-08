# FLINT

* This repository contains implementation for the paper:  ["A Framework to Learn with Interpretation"](https://arxiv.org/abs/2010.09345) by *Jayneel Parekh, Pavlo Mozharovskyi, Florence d'Alché-Buc* (accepted at NeurIPS 2021).

* This repo is currently being updated. Please visit back in a few days 

### Setup

Install a new conda environment with the ```env_minimal.yml``` file. You can start with [miniconda installation](https://docs.conda.io/en/latest/miniconda.html) if you are completely unfamiliar with anaconda   
```sh
conda env create -f env_minimal.yml
conda activate flint
```

### Usage

```
Command: python flint.py [mode] [dataset] [use-gpu] [model_name]

Functionality available: 
(a) mode: Training or testing phase (Options: [train , test])

(b) dataset: Name of Dataset (Options: [mnist, fmnist, qdraw, cifar10, cub])
    
(c) use-gpu: To use GPU or CPU for computation (Options: [True, False])

(d) model_name: If mode=test, takes in the name of model to be analyzed.

Example : python flint.py test qdraw False v18_if5_cd_0.30_run1.pt
```


### Pre-trained models


