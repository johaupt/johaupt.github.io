---
layout: post
title:  "Linear Regression with Stochastic Gradient Descent in Pytorch"
date:   2019-03-02
categories:
  - pytorch
  - tutorial
  - python
  - neural network
---


# Linear Regression with Pytorch

## Data and data loading


```python
import numpy as np
import pandas as pd
```


```python
def generate_experiment(n_obs,n_var):
    # Draw X randomly
    X = np.random.multivariate_normal(
             np.zeros(n_var),
             np.eye(n_var),
             n_obs
             )

    # Draw some coefficients
    beta = np.random.normal(loc=0., scale=1, size=n_var)

    y = np.dot(X,beta) + np.random.normal(loc=0., scale=0.1, size=n_obs)

    return X, y, beta


```


```python
X, y, coef = generate_experiment(n_obs = 1000, n_var=4)
```

Data loading in pytorch is the infrastructure that passes a mini-batch of the data to the training loop. It requires two pieces:
- A DataLoader handles the sampling and requests the indices of observations from the data
- The Dataset defines a class with functions \__len\__() to tell the DataLoader how many observations are available and \__getitem\__() to collect specific observations from the indices requested by the DataLoader. The Dataset can either contain the data (e.g. as numpy arrays) or contain the procedure to load them when requested (e.g. from an image folder).



```python
from torch.utils.data import Dataset, DataLoader
```


```python
class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return X.shape[0]

    def __getitem__(self, idx):
        return X[idx,:], y[idx]

```


```python
data = ExperimentData(X,y)
```


```python
data.X.shape, data.y.shape
```




    ((1000, 4), (1000,))



The batch size determines how many observations are passed to the model before updating. More observations give a better signal how to update the parameters (less noisy gradient), but take more time to calculate so updating is slower.


```python
data_loader = DataLoader(data, batch_size=128, shuffle=False, )
```

## Defining the model


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
```

**Step 1: Architecture I: The model architecture**    
Define all the layers that make up the network. The simple regression model is equivalent to a single (not a hidden!) linear layer that takes the weighted inputs, adds a bias and returns the result.

There is one design choice to make. Everything that has parameters need to be defined during initialization (since the weights need to be safed within the object), but functional activations like ReLU can also happen during the forward process. I our case, we don't have any non-linear activation functions like in a 'real' neural network, so we don't care.

The super() function in super().\__init\__() doesn't need arguments in Python 3, but you'll often see super(model, self).\__init\__()

**Step 2: Architecture II: The forward pass**    
Specify what happens when data is passed to the model. We defined the pieces (i.e. layers) above, but the flow of the model is still missing. This leaves us a lot of flexibility, which we really don't need for the simple regression model.    
Pass the data x to the linear layer and return the result.


```python
class regression(nn.Module):
    def __init__(self, input_dim):
        # Applies the init method from the parent class nn.Module
        # Lots of backend and definitions that all neural networks modules use
        # Check it out here:
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
        super().__init__()

        # Safe the input in case we want to use/see it
        self.input_dim = input_dim

        # One layer
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred     

```

**Step 3: Model fitting**    
See comments within the code below

It's critical to remember to **unsqueeze the true value or label** if is a numpy vector (shape [n.]). Pytorch fails silently when calculating the MSE on y (shape [n,]) and y_pred (shape [n,1]) returning errors of dimension [n,n], which in turn gets reduced to a scalar MSE by .mean(). This one took me a long time to figure out myself...


```python
def fit(model, data_loader, optim, epochs):
    # Iterate over the data a number of times
    for epoch in range(epochs):
        # For each subset of the data sampled by data_loader
        for i, (X,y) in enumerate(data_loader):

            # Make sure data is float and not, for example, integers.
            # Wrong format will cause an error during computation.
            X = X.float()
            y = y.unsqueeze(1).float()

            # Make the data a Variable over which gradients can
            # automatically be computed
            X = Variable(X, requires_grad=True)
            y = Variable(y, requires_grad=True)

            # Make a prediction for the input X
            pred = model(X)

            # Calculate the mean squared error loss
            # Pytorch naturally has a class for that
            loss = (y-pred).pow(2).mean()

            # Throw away the gradients that were previously
            # computed and stored in the optimizer
            optim.zero_grad()
            # Calculate the gradient for all parts of the above
            # calculation, also called a 'backward' pass
            loss.backward()
            # Update the weights as specified in the optimizer
            # Here: Gradient descent update with learning rate as
            #       specified
            optim.step()

        # Give some feedback after each 5th pass through the data
        if epoch % 5 == 0:
            print(f"loss: {loss}")

    return None
```

## Inference

These are the true coefficients used to generate the data. We only know them, because we simulated the dataset, of course.


```python
coef
```




    array([ 1.10270833, -0.39130098, -0.32453641,  1.18107399])



Let's see what performance we can expect by using the LinearRegression from scikit-learn.


```python
from sklearn.metrics import mean_absolute_error
```


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data.X,data.y)
print(model.coef_)
mean_absolute_error(y, model.predict(X))
```

    [ 1.10391649 -0.38996406 -0.32505661  1.18590587]





    0.0817587002716065



The data is created by a linear model, so the linear regression is able to capture the coefficients almost exactly and reduce the prediction error almost to 0 (e-15 means 0 followed by 14 zeros then the number in the front).

Let's build our pytorch regression model. It has 4 inputs, only one linear layer and one output. We include a *bias* term, which is equivalent to the *constant* in statistic parlance.


```python
regnet = regression(input_dim=4)
```


```python
regnet
```




    regression(
      (linear): Linear(in_features=4, out_features=1, bias=True)
    )



The weights aka coefficients are initialized randomly by default and we can look at them using function parameters().


```python
list(regnet.parameters())
```




    [Parameter containing:
     tensor([[-0.4913,  0.1847, -0.1824,  0.4594]], requires_grad=True),
     Parameter containing:
     tensor([0.4987], requires_grad=True)]



We use the same function to tell the optimizer which weights we want to update based on the data. For simplicity, I use plain stochastic gradient descent.     
The learning rate is the step size at which parameters are updated. It's typically somewhere around [0.1;0.00001], but is connected to the loss function and batch size, so it's one of the parameters that we can tune. I tuned it manually aka played around with it a bit until results looked good.


```python
from torch.optim import SGD, Adam
```


```python
optim = SGD(regnet.parameters(), lr=0.01)
```


```python
fit(regnet,
    data_loader,
    optim = optim,
    epochs = 100)
```

    loss: 3.4608473777770996
    loss: 0.7528247237205505
    loss: 0.16800978779792786
    loss: 0.04239470139145851
    loss: 0.01585623063147068
    loss: 0.010487892664968967
    loss: 0.009523425251245499
    loss: 0.009412898682057858
    loss: 0.00943633820861578
    loss: 0.009463834576308727
    loss: 0.009480305947363377
    loss: 0.009488780982792377
    loss: 0.009492901153862476
    loss: 0.009494856931269169
    loss: 0.009495782665908337
    loss: 0.009496204555034637
    loss: 0.009496405720710754
    loss: 0.009496488608419895
    loss: 0.009496539831161499
    loss: 0.009496556594967842


For testing purposes, the code below would be a way to set the true weights manually and see if prediction works.


```python
#regnet.linear.weight.data = torch.Tensor([coef])
#regnet.linear.bias.data = torch.Tensor([0.,])
```

Let's check the estimated parameters against the results from the sklearn linear regression model


```python
model.coef_
```




    array([ 1.10391649, -0.38996406, -0.32505661,  1.18590587])




```python
list(regnet.parameters())
```




    [Parameter containing:
     tensor([[ 1.1035, -0.3897, -0.3244,  1.1858]], requires_grad=True),
     Parameter containing:
     tensor([0.0020], requires_grad=True)]



To make a prediction we need to pass a tensor to the network, something the data loader to care of for us during training.


```python
pred = regnet(torch.Tensor(X).float())
```


```python
pred[:5]
```




    tensor([[-1.0094],
            [-0.2464],
            [ 2.2665],
            [-1.6032],
            [ 1.2057]], grad_fn=<SliceBackward>)



The predictions are still a tensor and we can't pass these on to an evaluation function. In order to transform them to a numpy array, we need to detach() them from the computational graph that pytorch creates in the background (so that it can do automatic differentiation).     
Then we can transform the tensor to a numpy array via one of its methods.


```python
pred = pred.detach().numpy()
```


```python
pred[:5].flatten()
```




    array([-1.0094117 , -0.24639717,  2.2664652 , -1.6031841 ,  1.2056617 ],
          dtype=float32)




```python
y[:5]
```




    array([-0.99319348, -0.26379455,  2.10439925, -1.72001673,  1.35644099])



Finally, the mean absolute error is almost as good as for the regression model optimized by numpy. Why not always exactly as good? There are better optimizers for simple regression than stochastic gradient descent. It's another story for neural networks, where we haven't found a better optimizer (yet), although there are some tweaks to SGD that are popular.


```python
mean_absolute_error(y_true=y, y_pred=pred)
```




    0.08176263854043513
