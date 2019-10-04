---
layout: post
title:  "Pytorch Implementation of Cross aka Interaction Layers: Cross and Deep Network"
date:   2019-03-22
categories:
  - python
  - pytorch
  - neural network
  - NN architecture
  - paper summary
---


# Cross layer for variable interaction

Pytorch implementation of the Cross Layer described in "Deep & Cross Network for Ad Click Predictions" (Wang, Fu, Fu & Wang, 2017): https://arxiv.org/abs/1708.05123

A typical extension to (generalized) linear regression models is to include variable interactions when we expect non-linear effects. For example,
$$y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x^2_{1i} + \beta_3 x_{2i} + \beta_4 x_{2i} x_{1i}$$
models an outcome variable that is non-linear in $$x_1$$ and where the effect of $$x_2$$ could have a moderating effect on the effect of $$x_1$$ and vice versa. But note how the interactions increase the dimensionality/complexity.

The idea behind cross layers is similar. In principle, a deep network should be able to learn variable interactions as needed. But the guideline is always if we can make the model more expressive by encoding more information, the network will have an easier time learning. 

In their paper, Ruoxi, Bin, Gang und Mingliang propose a module that calculates variable interactions efficiently. The efficiency comes from reducing the matrix of interactions through a weighted sum with trained weights, to which they also add a bias term.  

In particular, we calculate the interactions between the input $$x_0$$ and current hidden layer of the cross module $x_l$. We reduce this interaction, which has one additional dimension due to the outer product, back to the orginal dimensionality by the weighted sum $$w_l$$. We then add the original current hidden layer back to its interaction with the input $$x_0$$. 

$$x_{l+1} = x_0x_l^{\top} w_l + b_l + x_l = f(x_l, w_l, b_l) + x_l$$



```python
import math
import numpy as np 

import torch
import torch.nn as nn
from torch.nn import Linear
```


```python
x0 = torch.tensor([[1, 2, 3], [0, 1, 2], [2,3,2],[10, 20, 30]]).float()
x0
```




    tensor([[ 1.,  2.,  3.],
            [ 0.,  1.,  2.],
            [ 2.,  3.,  2.],
            [10., 20., 30.]])



The outer product for matrices was hard for me to imagine. That's a math rather than a machine learning problem, but it made coding the layer difficult, so I'll walk through it.

Just like with a vector outer product, the outer product for two matrices extends the dimensionality by 1, so the output is a 3D tensor.

Extend by one dimension? Let's see it for vectors:


```python
# 1D input
a = torch.tensor([1,2,3])
# 2D output
np.outer(a,a)
```




    array([[1, 2, 3],
           [2, 4, 6],
           [3, 6, 9]])



Pytorch doesn't have a function .outer(), so how could we do an outer product? We make the vectors into matrices first and multiply those! A vertical vector to the left and a horizontal vector on top is also how I would draw the outer product on paper to explain how it works, so seeing the matrix expansion helped deepen my understanding of matrix multiplication in general. Nice.


```python
# First vector is vertical by adding a column dimension 1
# unsqueeze() adds a dimension after some dimension i
# -1 means first dimension starting from the last as usual
a.unsqueeze(-1)
```




    tensor([[1],
            [2],
            [3]])




```python
# Second vector is horizontal by adding a row dimension 1
a.unsqueeze(-2)
```




    tensor([[1, 2, 3]])




```python
torch.mm(a.unsqueeze(-1), a.unsqueeze(-2))
```




    tensor([[1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]])



Now the same concept, but for matrices. Add an additional dimension, then do multiplication. In this case, not matrix multiplication but batch matrix multiplication, since we have more than one matrix.

See how each matrix is the interaction output for one observation. $$x_i^2$$ on the diagonal and $$x_i \cdot x_j \; \forall i,j$$ in the upper and lower triangle.


```python
x0xl = torch.bmm(x0.unsqueeze(-1), x0.unsqueeze(-2))
x0xl
```




    tensor([[[  1.,   2.,   3.],
             [  2.,   4.,   6.],
             [  3.,   6.,   9.]],
    
            [[  0.,   0.,   0.],
             [  0.,   1.,   2.],
             [  0.,   2.,   4.]],
    
            [[  4.,   6.,   4.],
             [  6.,   9.,   6.],
             [  4.,   6.,   4.]],
    
            [[100., 200., 300.],
             [200., 400., 600.],
             [300., 600., 900.]]])



Reduce the outer product by taking a weighted sum over the rows. The weights are the trainable weights of the linear layer and we add the trainable bias of the hidden layer. To see the weighted average more clearly, I set the weights to 1 and the bias to 0.


```python
weights = torch.ones(x0.shape[1])
bias = torch.zeros(x0.shape[1])
```


```python
torch.tensordot(x0xl, weights, dims=[[-1],[0]]) + bias
```




    tensor([[   6.,   12.,   18.],
            [   0.,    3.,    6.],
            [  14.,   21.,   14.],
            [ 600., 1200., 1800.]])



Add a shortcut connection from the original input $$x_0$$ to each layer as in the residual network by adding the input to the output at each layer.


```python
torch.tensordot(x0xl, weights, dims=[[-1],[0]]) + bias + x0
```




    tensor([[   7.,   14.,   21.],
            [   0.,    4.,    8.],
            [  16.,   24.,   16.],
            [ 610., 1220., 1830.]])



Helpful implementation in tensorflow: 
https://www.dropbox.com/sh/11c5mo0f7g42wgs/AACD73aZwl9NkcWnXkMTQX2Xa/Code?dl=0&preview=model.py&subfolder_nav_tracking=1

Here is the whole thing wrapped up in a custom module. 


```python
class Cross(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features
        
        self.weights = nn.Parameter(torch.Tensor(input_features))
        # Kaiming/He initialization with a=0
        #nn.init.normal_(self.weights, mean=0, std=math.sqrt(2/input_features))
        nn.init.constant_(self.weights, 1.)
        
        self.bias = nn.Parameter(torch.Tensor(input_features))
        nn.init.constant_(self.bias, 0.)
        
    def forward(self, x0, x):
        x0xl = torch.bmm(x0.unsqueeze(-1), x.unsqueeze(-2))
        return torch.tensordot(x0xl, self.weights, [[-1],[0]]) + self.bias + x
    
    # Define some output to give when layer 
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.input_features
        )
```


```python
cross = Cross( x0.shape[1] )
```


```python
cross(x0,x0)
```




    tensor([[   7.,   14.,   21.],
            [   0.,    4.,    8.],
            [  16.,   24.,   16.],
            [ 610., 1220., 1830.]], grad_fn=<AddBackward0>)




```python
list(cross.parameters())
```




    [Parameter containing:
     tensor([1., 1., 1.], requires_grad=True), Parameter containing:
     tensor([0., 0., 0.], requires_grad=True)]


