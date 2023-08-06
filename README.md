# PyTorch Implementation of the LCA Sparse Coding Algorithm

[![CI](https://github.com/lanl/lca-pytorch/actions/workflows/build.yml/badge.svg)](https://github.com/lanl/lca-pytorch/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/lanl/lca-pytorch/branch/main/graph/badge.svg?token=XfMW3nzzj0)](https://codecov.io/gh/lanl/lca-pytorch)
![CodeQL](https://github.com/lanl/lca-pytorch/workflows/CodeQL/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

LCA-PyTorch (lcapt) provides the ability to flexibly build single- or multi-layer convolutional sparse coding networks in PyTorch with the [Locally Competitive Algorithm (LCA)](https://bpb-us-e1.wpmucdn.com/blogs.rice.edu/dist/c/3448/files/2014/07/neco2008.pdf). LCA-Pytorch currently supports 1D, 2D, and 3D convolutional LCA layers, which maintain all the functionality and behavior of PyTorch convolutional layers. We currently do not support Linear (a.k.a. fully-connected) layers, but it is possible to implement the equivalent of a Linear layer with convolutions.

## Installation  

### Dependencies  

Required:
* Python (>= 3.8)

Recommended:
* GPU(s) with NVIDIA CUDA (>= 11.0) and NVIDIA cuDNN (>= v7)

### Pip Installation

```
pip install git+https://github.com/lanl/lca-pytorch.git
```

### Manual Installation

```
git clone git@github.com:lanl/lca-pytorch.git
cd lca-pytorch
pip install .
```

## Usage  

LCA-PyTorch layers inherit all functionality of standard PyTorch layers.

```python
import torch
import torch.nn as nn

from lcapt.lca import LCAConv2D

# create a dummy input
inputs = torch.zeros(1, 3, 32, 32)

# 2D conv layer in PyTorch
pt_conv = nn.Conv2d(
  in_channels=3,
  out_channels=64,
  kernel_size=7,
  stride=2,
  padding=3
)
pt_out = pt_conv(inputs)

# 2D conv layer in LCA-PyTorch
lcapt_conv = LCAConv2D(
  out_neurons=64,
  in_neurons=3,
  kernel_size=7,
  stride=2,
  pad='same'
)
lcapt_out = lcapt_conv(inputs)
```

## Locally Competitive Algorithm (LCA)

LCA solves the $\ell_1$-penalized reconstruction problem

<div align="center">

$\underset{a}\min \lvert|s - a * \Phi \rvert|_2^2 + \lambda \lvert| a \rvert|_1$

<div align="left">

where $s$ is an input, $a$ is a sparse (i.e. mostly zeros) representation of $s$, $*$ is the convolution operation, $\Phi$ is a dictionary of convolutional features, $a * \Phi$ is the reconstruction of $s$, and $\lambda$ determines the tradeoff between reconstruction fidelity and sparsity in $a$. The equation above is convex in $a$, and LCA solves it by implementing a dynamical system of leaky integrate-and-fire neurons

<div align="center">

$\dot{u}(t) = \frac{1}{\tau} \big[b(t) - u(t) - a(t) * G \big]$

<div align="left">

in which each neuron's membrane potential, $u(t)$, is charged up or down by the bottom-up drive from the stimulus, $b(t) = s(t) * \Phi$ and is leaky via the term $-u(t)$. $u(t)$ can also be inhibited or excited by active surrounding neurons via the term $-a(t) * G$, where $a(t)=\Gamma_\lambda (u(t))$ is the neuron's activation computed by applying a firing threshold $\lambda$ to $u(t)$, and $G=\Phi * \Phi - I$. This means that a given neuron will modulate a neighboring neuron in proportion to the similarity between their receptive fields and how active it is at that time.

Below is a mapping between the variable names used in this implementation and those used in [Rozell et al.'s formulation](https://bpb-us-e1.wpmucdn.com/blogs.rice.edu/dist/c/3448/files/2014/07/neco2008.pdf) of LCA.

<div align="center">

| **LCA-PyTorch Variable** | **Rozell Variable** | **Description** |
| --- | --- | --- |
| input_drive | $b(t)$ | Drive from the inputs/stimulus |
| states | $u(t)$ | Internal state/membrane potential |
| acts | $a(t)$ | Code/Representation/External Communication |
| lambda_ | $\lambda$ | Transfer function threshold value |
| weights | $\Phi$ | Dictionary/Features |
| inputs | $s(t)$ | Input data |
| recons | $\hat{s}(t)$ | Reconstruction of the input |
| tau | $\tau$ | LCA time constant |

</div>

## Examples

  * Dictionary Learning Using Built-In Update Method
    * [Dictionary Learning on Cifar-10 Images](https://github.com/lanl/lca-pytorch/blob/main/examples/builtin_dictionary_learning_cifar.ipynb)
    * [Fully-Connected Dictionary Learning on MNIST](https://github.com/lanl/lca-pytorch/blob/main/examples/builtin_dictionary_learning_mnist_fc.ipynb)
  
  * Dictionary Learning Using PyTorch Optimizer  
    * [Dictionary Learning on Cifar-10 Images](https://github.com/lanl/lca-pytorch/blob/main/examples/pytorch_optim_dictionary_learning_cifar.ipynb)

## License
LCA-PyTorch is provided under a BSD license with a "modifications must be indicated" clause.  See [the LICENSE file](https://github.com/lanl/lca-pytorch/blob/main/LICENSE) for the full text. Internally, the LCA-PyTorch package is known as LA-CC-23-064.
