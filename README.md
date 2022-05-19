# FLINT

* This repository contains implementation for the paper:  ["A Framework to Learn with Interpretation"](https://arxiv.org/abs/2010.09345) by *Jayneel Parekh, Pavlo Mozharovskyi, Florence d'Alché-Buc* (Presented at NeurIPS 2021).

### Setup

Install a new conda environment with the ```env_minimal.yml``` file. You can start with [miniconda installation](https://docs.conda.io/en/latest/miniconda.html) if you are completely unfamiliar with anaconda   
```sh
conda env create -f env_minimal.yml
conda activate flint
```


### Pre-trained models

The pre-trained network files for CIFAR-10, QuickDraw and CUB-200 can be downloaded from this drive link: https://drive.google.com/file/d/15RjZUlIuW5JF2pz5ClKvpOp7uN-cOn3y/view?usp=sharing

The folder when extracted will contain three network weights file. For using any file you need to place it in appropriate file directory of your code according to your dataset. For eg. if your dataset is 'qdraw' than the right subfolder for the network file is: flint_home_dir/output/qdraw_output/



### Usage

```
Command: python flint.py [mode] [dataset] [use-gpu] [model_name]

Functionality available: 
(a) mode: Training or testing phase (Options: [train , test])

(b) dataset: Name of Dataset (Options: [mnist, fmnist, qdraw, cifar10, cub])
    
(c) use-gpu: To use GPU or CPU for computation (Options: [True, False])

(d) model_name: If mode=test, takes in the name of model to be analyzed.

Example : python flint.py test qdraw False trained_qdraw.pt
```
