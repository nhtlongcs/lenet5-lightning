# LetNet Pytorch Lightning

This work is an unofficial implementation of LeNet5 on MNIST dataset, which is computer vision coursework.

Before using this repo, please install the following packages by running the following commands:

```bash
conda env create -f environment.yml
pip install -e .
```

Or if you are using a colab (cannot install by conda), you can install the packages by running the following commands:

```bash
pip install -r requirements.txt
pip install -e .
```

[Wandb](https://wandb.ai/nhtlong/vnu-tgmt)

</p>
<a href="https://colab.research.google.com/github/nhtlongcs/lenet5-lightning/blob/master/notebook/LeNet5.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Inference

I provide simple scripts for training and testing purpose.

```bash
$ ./tools/train.sh
$ ./tools/test.sh <path-to-ckpt>
```
