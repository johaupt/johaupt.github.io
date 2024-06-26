---
layout: post
category: blog
title:  "Recalibration after Negative Sampling"
date:   2021-06-02
tags:
- calibration
- sampling
- causal machine learning
---

_This post shows the transformation function that calibrates predictions from a model trained on downsampled data back to the original data. We'll get an intuition, do the derivation of the transformation, and a scikit-learn example with simulated data._

## Negative Sampling

Real-time bidding and natural language processing pre-training are two applications were positive cases are much rarer in the data than negative cases. As an example, there are thousands users who don't click on an ad for each user who does, an even fewer who buy something after the click compared to every user who just ignored it.

When negative cases are so abundant and overall data size is large, it can be helpful to ignore some or even many of the negative cases during model training because they don't provide much additional information but slow down model estimation because they need to be processed. 

A common way to do that is by randomly sampling just a part of the negative cases, while keeping all positive cases in the data. Effectively throwing away negative cases at random decreases the data size, so model estimation generally speeds up. However, by changing the ratio between positive and negative cases, any model estimated on the data (without any correction) will overestimate the probability of positive cases occuring. While we can reasonably expect the model to do distinguish between positive and negative cases equally well (e.g. the AUC doesn't change), we should expect it to be badly calibrated (e.g. high average probability for positive cases even though they are overall very rare and log-loss decreases on test data which is not sampled).

It is possible to re-calibrate the model predictions to match the original data before resampling. For convenience, I'll denote the baserate of positive cases $P(y=1)$ in the original data by $b$ and in the re-sampled data by $b'$. The model predictions based on the resampled training data are $p'$, while the predictions that are calibrated for the original data and that we want to use in practice are $p$.

The formulas to know are

1. The **rate at which we sample negative cases $w$**, which we may empirically optimize or pick to match a particular target ratio of positive vs. negative cases (absent further information often 50:50)
$$ w = \frac{\text{original ratio}}{\text{target ratio}} = \frac{b}{1-b} \cdot \frac{1-b'}{b'} $$

2. The **transformation for the model predictions $p'$** which recalibrates them to calibrated predictions $p$ that match the average occurence rate in the original data
$$p = \frac{p'}{p'+\frac{1-p'}{w}} $$

The rest of the post shows how we derive these formulas and what the assumptions are on the way and then shows an example with scikit-learn.

## Intuition
We'll start with a sloppy intuition of how to correct change the estimate on resampled data to match the unsampled data. Imagine a simple case: You've downsampled the negative class keeping each negative case with $p=0.5$, then trained a decision tree. Now zoom into any one end leaf. The prediction function $\hat{p}$ on the data in the leaf $L$ is the number positive class cases $N_1$ in the leaf $N_1^L$ divided by the total number of cases.

$$
\begin{align*} 
\hat{p} &= \frac{N^L_1}{N^L_1+N^L_0}
\end{align*}
$$

Imagine we have a leaf with 10 positive cases and 90 negative cases during training, so we are working with the resampled data. What would the leaf look like if we sent the unsampled data through the tree?
I would expect the same number of positive cases, since we didn't touch those. I would expect 50% more negative cases. We removed 50% of negative cases during downsampling and since we removed them randomly I would expect roughly 50% of negative cases missing no matter which leaf I am looking at. 

Since I know how many negative cases I removed, we could fix the number of negative cases in the prediction function from the actual 90 in the resampled data to the expected 180 in the unsampled data. We double with factor 2 because we removed half of the negative cases.

$$
\begin{align*} 
\hat{p}^L &= \frac{10}{10+ 90 \cdot 2} \\
\hat{p}^L &= \frac{10}{10+ 90 \cdot 1/50\%} 
\end{align*}
$$


That's it, we could use this as a new prediction function within our tree.

As food for thought, note that I used the downsampling ratio for the full data for the correction $N'_0/N_0$, not the empirical ratio in the leaf $N'^L_0/N^L_0$ which I could get by passing the unsampled data through the tree splits. These are the same in expectation but the actual ratio in the leaf will be different due to the randomness of the sampling. 

Practically though, implementations often don't expose the level of prediction function or implement class weights. Instead, what if we had only the predictions from the model trained on the downsampled data $p'$?

Let's try to expand our formula above to turn the counts into probabilities. I use $N'_0$ as count of negative cases in the sampled data.

$$
\begin{align*} 
\hat{p}^L &= \frac{N^L_1}{N^L_1+ N^L_0 \cdot \frac{N_0}{N'_0}} \quad \text{divide by total count}\\
\hat{p}^L &= \frac{\frac{N^L_1}{N^L}}{\frac{N^L_1}{N^L}+ \frac{N^L_0}{N^L} \cdot \frac{N_0}{N'_0}}\\
\hat{p}^L &= \frac{p^L_1}{p^L_1+ p^L_0 \cdot \frac{N_0/N_1}{N'_0/N_1}} \quad \text{note: } N_1=N'_1\\
\hat{p}^L &= \frac{p^L_1}{p^L_1+ p^L_0 \cdot \frac{N_0/N_1}{N'_0/N'_1}}
\end{align*}
$$

This is the same as the common formula after some rearrangement that makes the correction less intuitive

$$
\begin{align*} 
w & = \frac{N_1/N}{N_0/N} \cdot \frac{N'_0/N'}{N'_1/N'} \\
&= \frac{N_1}{N_0} \cdot \frac{N'_0}{N'_1} \\
&= \left(\frac{N_0}{N_1} \cdot \frac{N'_1}{N'_0} \right)^{-1}\\
\end{align*}
$$

## Derivation of the Re-Calibration Formula

A lot of the research around resampling comes from the literature around cost-sensitive learning. Elkan (2001) The Foundations of Cost-Sensitive Learning summarizes the reasoning well and shows many of the, well, foundations in some detail. In that literature, people aim to sample the data to calibrate model predictions so that a 0.5 decision  threshold on the probability matches the decision context, made up by the benefits and costs of correct and incorrect decisions. I'll ignore the cost-informed decision-making background, but their research on resampling data is still helpful and the following derivations are informed by Elkan (2001). 

Start with Bayes theorem

$$
\begin{align*} 
P(y=1|x) &= \frac{P(x|y=1)P(y=1)}{P(x)} \quad \text{expand P(x) for all (two) classes of y}\\
 &= \frac{P(x|y=1)P(y=1)}{P(x|y=1)P(y=1) + P(x|y=0)P(y=0) }  \quad \text{rearrange to}\\
 &= \frac{P(y=1)}{P(y=1)+\frac{P(x|y=0)}{P(x|y=1)} (1-P(y=1)) }
\end{align*}
$$

Or with our simplified notation

$$p = \frac{b}{b +\frac{P(x|y=0)}{P(x|y=1)} (1-b) }$$

We randomly sample some of the observations for which $y=0$ to create the target, downsampled dataset. Within this dataset, the probability of $y$ given a set of features $x$ changes (there are now more $y=1$ after all, but the distribution of $X$ given $Y$ doesn't change. Hence $P(y=1\|x)$ and $P(y=0\|x)$ are different but $P(x\|y=1)$ and $P(x\|y=0)$ are the same as in the original dataset and we can use this for

$$ p' = \frac{b'}{b' +\frac{P(x|y=0)}{P(x|y=1)} (1-b') } $$

We can join these to expression by inserting $\frac{P(x\|y=0)}{P(x\|y=1)}$ into the expression for $p$. First we reformulate $p'$.

$$
\begin{align*}
\frac{1}{p'} &= \frac{b' +\frac{P(x|y=0)}{P(x|y=1)} (1-b') }{b'}\\
\frac{b'}{p'} -b' &= \frac{P(x|y=0)}{P(x|y=1)} (1-b') \\
\frac{b'-bp'}{p'(1-b')}  &= \frac{P(x|y=0)}{P(x|y=1)}  \\
\end{align*}
$$

Then we replace $\frac{P(x\|y=0)}{P(x\|y=1)}$ in the equation for $p$

$$
\begin{align*}
p &= \frac{b}{b +\frac{P(x|y=0)}{P(x|y=1)} (1-b) } \\
&= \frac{b}{b +\frac{b'-bp'}{p'(1-b')} (1-b) } \quad \text{rearrange}\\
&= \frac{b}{b +\frac{b'}{(1-b')} \frac{(1-p')}{p'} (1-b) } \\
&= \frac{b}{b +\frac{b'}{(1-b')} \frac{(1-p')}{p'} \frac{(1-b)b}{b} } \quad \text{insert } \frac{1}{w}=\frac{b'}{(1-b')} \frac{(1-b)}{b} \\
&= \frac{b}{b + \frac{b}{w}\frac{(1-p')}{p'}  } \\
&= \frac{1}{1 + \frac{1}{w}\frac{(1-p')}{p'}  } \\
&= \frac{p'}{p' + \frac{(1-p')}{w}  }
\end{align*}
$$

This is the recalibration formula from above and we see that it works for any downsampling ratio $w$ that we pick. It's common to downsample to a completely balanced training dataset, i.e. with a target ratio of 1:1, for which $w = \frac{b}{1-b} \frac{0.5}{0.5} = \frac{b}{1-b}$. In practice, less downsampling often gives slightly better models because less data is discarded. 

## Example


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
```


```python
X, y = make_classification(n_samples=10000, n_informative=10, weights=[0.9])
```

Here we have some imbalance with a baseline probability for $y$ that is 10%. 


```python
y.mean()
```




    0.103



### Model without downsampling


```python
logit = LogisticRegression(penalty='none').fit(X,y)
```


```python
pred = logit.predict_proba(X)[:,1]
```


```python
pred.mean()
```




    0.10299859038609889



Let's also have a look at the calibration curve of the predictions to be able to compare later. Logistic regression is usually calibrated well, which is not the case for all machine learning models.


```python
fraction_of_positives, mean_predicted_value = calibration_curve(y, pred, n_bins=5)
```


```python
np.quantile(pred, [0,0.5, 0.75, 0.9, 0.95, 1])
```




    array([3.78453590e-06, 9.83524180e-03, 4.75441847e-02, 3.33898140e-01,
           8.51368277e-01, 9.99970943e-01])




```python
plt.plot(mean_predicted_value, fraction_of_positives)
plt.plot([0,1],[0,1]);
```


![png](/assets/img/downsampling_recalibration/output_26_0.png)


### Model with Downsampling

Downsample the negative cases by a negative sampling rate of 1/9 for a target ratio of 1:1. A ratio of 1:1 is not necessarily the best target ratio in practice, but it is salient because it is "balanced".


```python
w = 1/9
count_y0 = (y==0).sum()
```


```python
neg_sample_idx = np.random.default_rng().choice(
    count_y0, size=int(w*count_y0.sum()), 
    replace=False)
```


```python
neg_sample_idx = np.arange(int(w*(y==0).sum()))
```


```python
X_sampled = np.vstack([X[y==0][neg_sample_idx], X[y==1]])
y_sampled = np.concatenate([y[y==0][neg_sample_idx], y[y==1]])
```


```python
X_sampled.shape
```




    (2026, 20)




```python
y_sampled.shape
```




    (2026,)



The new ratio of positive cases is ~50% as expected.


```python
y_sampled.mean()
```




    0.508390918065153



This time we train the model on the downsampled data.


```python
logit = LogisticRegression(penalty='none').fit(X_sampled,y_sampled)
```


```python
pred = logit.predict_proba(X_sampled)[:,1]
```

Unsurprisingly, the average prediction is the rate of positive cases in the resampled data, which is much higher than in the original dataset.


```python
pred.mean()
```




    0.5083907547956312



We therefore want to use the formula derived above to re-calibrate the predictions to the original distribution of cases.


```python
def calibrate(p, w):
    return p / (p + ((1-p)/w))
```

Get the predictions of the model on the original, unsampled dataset.


```python
pred = logit.predict_proba(X)[:,1]
```

As expected, the model overpredicts the occurance of positive cases.


```python
pred.mean()
```




    0.23492324259220376




```python
pred2 = calibrate(pred,w)
```

After re-calibration, the average model prediction matches the ratio of positive cases almost exactly.


```python
pred2.mean()
```




    0.09856897573250073




```python
y.mean()
```




    0.103



The overall calibration looks also good as judged by the calibration curve of the predictions (close to the diagonal line), at least for the prediction range between 0 and 45% where we find most (95%) of our predictions.


```python
fraction_of_positives, mean_predicted_value = calibration_curve(y, pred2, n_bins=5)
```


```python
np.quantile(pred2, [0,0.5, 0.75, 0.9, 0.95, 1])
```




    array([1.18645827e-05, 1.07949077e-02, 5.11099944e-02, 3.16203654e-01,
           7.56029748e-01, 9.99710693e-01])




```python
plt.plot(mean_predicted_value, fraction_of_positives)
plt.plot([0,1],[0,1]);
```


![png](/assets/img/downsampling_recalibration/output_55_0.png)

