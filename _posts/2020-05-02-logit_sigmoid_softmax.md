---
layout: post
title:  "Sigmoid Functions and Logistic Regression"
date:   2020-05-02
categories:
    -summary
    -basics
---


*This post is a quick summary of the connection between the logistic link function, sigmoid activation, multinomial logistic regression and the softmax transformation. *

The term *sigmoid* function, or activation for neural networks, is slightly misleading because it describes a [shape of function](https://en.wikipedia.org/wiki/Sigmoid_function) rather than a specific function. Sigmoid translates to *sigma*-shaped, but the sigma in question is not $\sigma$ but the obscure $\varsigma$ which actually looks like a modern *S*. It might well be that many problems in machine learning are the result of mathematicians choosing naming conventions based on some obscure insider knowledge and computer scientists laughing along without getting the joke. 

The most popular sigmoid function is the *logistic function*, which in its general form looks like

$$f(x)={\frac {L}{1+e^{-k(x-x_{0})}}}$$

with 

$e$ = the natural logarithm base,    
$x_{0}$ = the $x$ value of the sigmoid's midpoint,    
$L$ = the curve's maximum value (for probabilities: 1)   
$k$ = the logistic growth rate or steepness of the curve.

For $L=1$, $x_0=0$ and $k=1$, the standard logistic function takes the form used in logistic regression, which is

$$f(x)={\frac {1}{1+e^{-x}}}$$

A different sigmoid function that is often used is the cumulative distribution function of the Gaussian/Normal distribution. The Gaussian CDF in place of the logistic function is called a *probit* link function for the generalized linear model. 

### Binary logistic regression and softmax

For more than two classes, logistic regression can be extended to multinomial logistic regression. Multinomial logistic regression is also equivalent to a final neural network layer with softmax activation. In fact, the logistic regression function can be derived from multinomial logistic regression/softmax for two classes.

For the [softmax activation](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), we divide the class prediction by the sum of all predictions to scale the output to [0;1] summing up to 1. For two classes and their respective probability $p_{0/1}$,

$$p_0 = \frac{e^{X\beta_0}}{e^{X\beta_0} + e^{X\beta_1}}\\
p_1 = \frac{e^{X\beta_1}}{e^{X\beta_0} + e^{X\beta_1}}$$

Because there are only two classes 0 and 1, we can simplify the formula:

\begin{align*}
p_1 &= \frac{e^{X\beta_1}}{e^{X\beta_0} + e^{X\beta_1}}\\
&= \frac{e^{X\beta_1}}{e^{X\beta_0} + e^{X\beta_1}} \cdot \frac{e^{X\beta_1}}{e^{X\beta_1}}\\
&= \frac{1}{\frac{e^{X\beta_0}}{e^{X\beta_1}} + 1}  && \big| \quad e^a/e^b = e^{a-b} \\
& = \frac{1}{e^{X\beta_0 - X\beta_1} +1} \\
& = \frac{1}{1+e^{X(\beta_0 -\beta_1)}} 
\end{align*}

Now if we define a parameter vector $\beta = - (\beta_1 - \beta_0)$, we get the classic logistic regression formula

$$ 
p_1 = \frac{1}{1+e^{-X\beta}} 
$$
