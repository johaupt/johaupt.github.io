---
layout: post
blog: stats
title:  "Logistic Regression as a Choice Model"
date:   2018-08-05
tags:
  - customer choice
  - logistic regression
  - statistical modeling
---

*Efficient summary how to motivate logistic regression and softmax regression as a (customer) choice model*.

Assume two choice alternatives: $$C_0$$ and $$C_1$$. A person picks at a choice point $$n$$ one of the alternatives *i,j* with higher utility $$U_{i/j}$$, which we define as a function of the characteristics $$x_{i/j}$$, weighted by some coefficients $$\beta_{i/j}$$ and an error $$\epsilon_{i/j,n}$$. Which choice model we derive depends on our assumption on the error term, e.g. *extreme value distribution* for the logit model or *normal distribution* for a probit model.

The probability to pick choice i is

\begin{align}
P_n(i|{i,j}) = P(U_{in} \geq U_{jn}) = P(V_{in} + \epsilon_{in} \geq V_{jn}+ \epsilon_{jn}) \text{,}
\end{align}

which is equivalent to 
\begin{align}
P_n(i|{i,j}) = P(\epsilon_{in} - \epsilon_{jn} \leq V_{in} - V_{jn}) \text{.}
\end{align}

So the the choice for an alternative happens when the unobserved influences do not outweight the deterministic advantage of the choice.

The difference between two random variables that follow a extreme value distribution follows a Logistic distribution (*Logit Choice Model*). Similarly, the difference between two random variables that follow a Gaussian distribution follows again a Gaussian distribution (*Probit Choice Model*). 

The cumulative distribution function of the Logistic function is 
\begin{align}
P(\epsilon \leq v) = \frac{1}{1+e^{-\frac{v - \mu}{s}}} \text{,}
\end{align}

where $$\mu$$ and $$s$$ are parameters of the error, which we assume to be 0 and 1, respectively. This choice seems to be justified since the scaling of utility functions is arbitrary and immune to linear scaling.

We can see that this is the equation from above with $$v = V_{in} - V_{jn}$$. We can expand the difference to put this in the form of 
\begin{align}
P(\epsilon \leq v) = \frac{1}{1+e^{-V_{in} - V_{jn}}} = \frac{1}{
1+\frac{
e^{V_{jn}}}{
e^{V_{in}}}
} = \frac{e^{V_{in}}} {e^{V_{in}} + e^{V_{jn}}}
\end{align}

Keep in mind that we are free to model $$V$$ however we want, for example by a non-linear neural network, but a linear model is the standard logit formulation:

\begin{align}
V_n = \beta_0 + x_n^{\top}\beta
\end{align}

When comparing to the no-choice option, e.g. will the customer on a webpage complete the purchase or abandon the cart, we can fix the one class to $$V_0=0$$ to get 

\begin{align}P(\epsilon \leq v) = \frac{1} {1 + e^{-V_{in}}} = \frac{1} {1 + e^{-(\beta_0 + x_n^{\top}\beta)}}\end{align} or alternatively

\begin{align} P(\epsilon \leq v) = \frac{e^{\beta_0 + x_n^{\top}\beta}} {1 + e^{\beta_0 + x_n^{\top}\beta}} \end{align}

We can estimate this by maximizing the likelihood over the data, for example using gradient ascent. But how to interpret the coefficients that we can derive?

We would need to get rid of the exponential function, but taking the log directly would leave us with another term $$\ln(1+e^{V_{in}})$$ from the denominator. To get rid of that, we could look divide by the probability for the second class and look at the ratio of the probabilities instead.

\begin{align}\frac{P(\epsilon \leq v)}{P(\epsilon \geq v)} = \frac{\frac{e^{\beta_0 + x_n^{\top}\beta}} {1 + e^{\beta_0 + x_n^{\top}\beta}}}{
\frac{1} {1 + e^{\beta_0 + x_n^{\top}\beta}}} = e^{\beta_0 + x_n^{\top}\beta} \text{.} \end{align}

Taking the natural log now gives us

\begin{align}\ln(\frac{P(C_1)}{P(C_0)}) = \beta_0 + x_n^{\top}\beta \text{,} \end{align}

which we know how to interpret, because that the standard regression formulation, although the outcome variable is the weird log odds ratio. For example, if $$\beta_k = 0.5$$, we know that a unit increase in $$x_k$$ increases the log odds by 0.5. What does that mean in terms of not-log odds?

$$\ln(odds)' - \ln(odds) = 0.5$$ is equivalent to

$$\frac{e^{\ln(odds'}}{e^{\ln(odds})} = e^{0.5}$$

$$\frac{odds'}{odds} = 1.649$$

$$odds' = 1.649 \cdot odds$$,

which translates to a ~65% increase in the odds. That does *not* translate to the statement, e.g. a customer who 35 years old is 65% more likely to buy than a customer who is 34. See [https://en.wikipedia.org/wiki/Odds](https://en.wikipedia.org/wiki/Odds) on how to interpret odds correctly.
