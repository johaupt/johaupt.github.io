---
layout: post
title:  "Causal Machine Learning: Individualized Treatment Effects and Uplift Modeling"
date:   2019-09-19
categories:
  - causal machine learning
---

*A comprehensive collection of state-of-the-art methods from causal machine learning or uplift modeling to estimate individualized treatment effects.*

# Table of Contents
0. [K-Model Approach](#k-model-approach)
    0. [Treatment residual specification](#treatment-residual-specification)
    0. [Multi-task network](#multi-task-network)
0. [Treatment Indicator as Variable](#treatment-indicator-as-variable)
    0. [Bayesian additive regression trees](#bayesian-additive-regression-trees)
0. [Outcome Transformation](#outcome-transformation)
    0. [Double robust estimation](#double-robust-estimation)
0. [Non-parametric Methods](#non-parametric-methods)
    0. [Causal tree](#causal-tree)
    0. [Pollienated transformed-outcome tree](#pollienated-transformed-outcome-tree)
    0. [Boosted causal trees](#boosted-causal-trees)
    0. [Generalized random forest](#generalized-random-forest)
0. [Bagged Causal MARS](#bagged-causal-multivariate-adaptive-regression-splines)
0. Modified Loss Function
    0. [Covariate transformation](#covariate-transformation)
    0. [R-learner](#r-learner)
0. [Estimated Treatment Effect Projection](#treatment-effect-projection)
    0. [X-learner](#x-learner)
0. [Benchmark Studies](#benchmark-studies)

# Motivation

Assume that we want to change the world. A little less grandiose, assume we want to take an action that will impact an outcome to be more like we prefer. A little more applied, fields like medicine and marketing make a decision to take action that increases life expectation by five years or makes grandma spend another dollar in the supermarket. To make that decision, we want to know in advance what effect the conceivable treatment will have on each individual/grandma. Remember treatment as a general term that can mean anything from a mail catalog to earlier starting time for a class to medication.     
Example questions and treatments are:

- How much more likely is a my middle-aged, sporty patient to be cured when given new medication as opposed to the standard medication?
- How much more likely are participants to complete the online training if we send them a progress report every week? Or are there certain participants that react negatively to perceived pressure?
- How much more will a customer buy if promised free shipping as opposed to not showing any banner?

Estimation of treatment effects on the individual level is a tricky problem, because we typically only have one shot at applying a treatment: either my grandma gets the coupon or she doesn't and we will be left to ponder the 'what if'. 'What if' type of questions belong to the domain of causal inference and estimating the treatment effect for individuals has become a task at the intersection of causal inference and machine learning. The same problem is known as heterogeneous treatment effects in social studies and medicine, conditional average treatment effects in econometrics and uplift modeling or prescriptive analytics in business intelligence. 

The fundamental problem of 'what if' is that we can only apply one treatment to each individual and observe their outcome. Since we don't know how the individual would have reacted under alternative treatment, we can never calculate the real treatment effect. What we can do is to look at a group of individuals who have received different treatments and the difference in their outcomes.

Research into estimating treatment effects from experiments and observational studies goes far into the last century, but the last ten years have seen renewed interest to leverage machine learning techniques to increase the power of known methods or build machine learning models specialized in estimating treatment effects.    
Unfortunately, research in different fields is fractured, with different termini and notation. The following list is meant as an incomplete, but comprehensive summary of state-of-the-art methods from bioinformatics, computer science, econometrics and business studies to estimate individualized treatment effects. I will assume that we know how randomized experiments work and that we have data on some individuals who have gotten treatment and some who haven't. 

# Notation
My notation is roughly in line with the econometric literature:

Covariates for individual *i* (*What we know about the person*   ): \\( X_i \\)  
(Estimated) Propensity score (*Their chance to get the coupon from us* ): \\( P(W=w|X), e(X)=P(T=1|X) \text{when} T \in {0;1} \\)    
Treatment Group Indicator (*If the person got a coupon or not*  ): \\(T\\)    
(Potential) Outcome *Y* for individual *i* under group assignment *T* (*If/How much they bought*): \\( Y_i(T)\\)     
(Estimated) Conditional outcome under group assignment *T* (*If/How much we think they should have bought*): \\( \mu(X_i, T_i) \\)     
(Estimated) Treatment Effect (*How much impact the coupon had*): \\( \tau_i \\) 

# K-Model approach
**(aka Two-Model Approach, T-Learner, Conditional Mean Regressions, Difference in Conditional Means)**    
The following approaches build a model to predict the outcome with and without treatment. By looking at the difference between the predicted outcomes, we can find out how much impact we can expect from the treatment.

When using the k-model approached, named after the number of models we need to train, we estimate an outcome model for each treatment group separately and calculate the treatment effect as the difference between the estimated outcomes.     
If there is one treatment, we will have two groups: treatment and control (where people might get nothing or Placebo). The impact of treatment vs. doing nothing is then the difference between the prediction of the model build on the treated and prediction of the model build on the control group.

The outcome models (*base learners*) can take any form, so we could use neural networks or boosted trees to do the heavy lifting. The downside of the k-model approach is that the outcome process may be difficult to estimate and that the errors of the two models in the difference may add up.

## Multi-task network
**(aka DragonNet)**    
We can of course use two neural networks as outcome models in the two-model framework. The two outcome models are likely very similar, since both approximate to a large extent the outcome process without treatment. We may be able to gain efficiency and improve calibration through parameter sharing in the lower hidden layers. The architecture is then best understood as a single multi-task network, with one loss calculated on the control group observations and one (or more) loss calculated on the treatment group observations.    

The multi-task architecture has an additional advantage when working with observational data. When we cannot conduct an experiment and treatment assignment is not random, we can correct for variables that impact the treatment assignment to still make unbiased estimates. It is in fact sufficient to correct only for the variables that impact treatment assignment (*propensity weighting*). An efficient way to filter the relevant information in the multi-task neural network is to correct the shared hidden layers. We correct the last shared layer, for example, by adding the treatment probability as an additional output. Predicting the treatment probability forces the hidden layers to distill the information that is necessary to predict treatment assignment and focus less on the information that is relevant only for outcome prediction, but doesn't differ between the control and treatment group.

*Shalit, U., Johansson, F. D., & Sontag, D. (2017). [Estimating individual treatment effect: generalization bounds and algorithms](https://arxiv.org/abs/1606.03976). Proceedings of the 34th International Conference on Machine Learning (ICML 2017).     
Shi, C., Blei, D. M., & Veitch, V. (2019). [Adapting Neural Networks for the Estimation of Treatment Effects](http://arxiv.org/abs/1906.02120). ArXiv:1906.02120 [Cs, Stat].     
Alaa, A. M., Weisz, M., & van der Schaar, M. (2017). [Deep Counterfactual Networks with Propensity-Dropout](http://arxiv.org/abs/1706.05966). ArXiv E-Prints, arXiv:1706.05966.*


## Treatment residual specification
We can improve model calibration in the two-model framework by 1) constructing estimates in the treatment group as an addition of the control and treatment model and 2) joint model training.     
To train a neural network that predicts the treatment effect directly, look at the observed outcomes under treatment as a sum of the outcome without treatment and the treatment effect. Then we could estimate one network that predicts the outcome without treatment for all observations and a second network that predicts the treatment effect that needs to be added for treated individuals, equivalent to the residual left by the outcome network for treated observations. Instead of two outcome models, this framework leaves us with one network that predicts the outcome and one network that predicts the treatment effect directly.
\\[
  Y = \text{nnet}_0 + T_i * \text{nnet}_1 
\\]
To ensure that the networks are in tune with each other, we should train them jointly. This does not require much effort: For each observation, we sum up the prediction of the outcome network and the prediction of the treatment network multiplied by the treatment indicator. We then backpropagate the error through both networks.     

*Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2017). [Bayesian Regression Tree Models for Causal Inference: Regularization, Confounding, and Heterogeneous Effects](https://doi.org/10.2139/ssrn.3048177). SSRN Electronic Journal.     
Farrell, M. H., Liang, T., & Misra, S. (2018). [Deep Neural Networks for Estimation and Inference: Application to Causal Effects and Other Semiparametric Estimands](https://arxiv.org/abs/1809.09953). ArXiv E-Prints, arXiv:1809.09953*


# Treatment Indicator as Variable 
**(aka Single-Model-Approach, S-Learner)**    

We can include the treatment indicator as a covariate into the model, optionally with interaction to other covariates. Predict the ITE via the difference of predicting an observation with treatment set to 0 and set to 1. Training a single model for the outcome is simple and often interpretable. 

A regression model with a linear additive treatment effect and interaction effects would look like this:
\\[
 Y = \beta_0 + \tau_0 D_i+ \tau D_i X_i + \beta X_i + \epsilon_i
\\]

Under linear regression, the interaction effects between all variables and the treatment indicator blow up the dimensionality quickly. Instead, we could use any machine learning model and include the treatment indicator as a variable. However, if the treatment effect is small relative to other effects on the outcome, then regularized machine learning methods may ignore the treatment variable completely.

The advantage of response model that include the treatment variable directly is that they are flexible when modeling multiple treatments or continuous treatments. We always train a single model. To predict the ITE, we feed in $X$ multiple times, each time with the treatment variable set to the value of interest. In the binary case, we would set the treatment variable once to 1 and once to 0 for the same $X$. The difference in predictions is the treatment effect estimate.

## Bayesian Additive Regression Trees
Use Bayesian Additive Regression Trees as the response model. The difference in posterior distributions gives a good estimate of the ITE in practice and gives an uncertainty estimate of the treatment effect that may be useful for risk-conscious decision-making or to do uncertainty based data collection. 

*Hill, J. L. (2011). [Bayesian Nonparametric Modeling for Causal Inference](https://doi.org/10.1198/jcgs.2010.08162). Journal of Computational and Graphical Statistics, 20(1), 217–240.*

# Outcome Transformation 
**(aka Modified Outcome Method, Class Variable Transformation, Generalized Weighted Uplift Method)**     
The outcome process (read: who completes their purchase in our online shop) is often more complicated than the process behind the treatment effect (read: who is impacted most by a coupon). The following approaches therefore aim to estimate the treatment effect directly without the need to build a good outcome model first.

Our job would be so much easier if we knew the actual treatment effect and could just train a regression model to predict it, but we can never know the actual treatment effect for an individual. However, we can find an artificial variable that is equal to the treatment effect in expectation.

The proxy variable is a transformation of the observed outcome for the individual:
\\[
Y^{TO}_i = T_i Y_i - (1-T_i) Y_i
\\]

Or including treatment propensity correction if we didn't do a 50:50 randomized experiment:
\\[
Y^{IPW}_i = T_i \cdot \frac{Y_i}{e(X_i)} - (1-T_i) \cdot \frac{Y_i}{1-e(X_i)}
\\]

The transformed outcome is a noisy but unbiased estimate of the treatment effect. As an unbiased estimate, it can be used as a target variable for model training. 
The transformed outcome can also be used to calculate a feasible estimate of the MSE between model estimate and true treatment effect that is useful for model comparison. 

*Hitsch, G. J., & Misra, S. (2018). [Heterogeneous Treatment Effects and Optimal Targeting Policy Evaluation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957). SSRN.*

## Double robust estimation 
The transformed outcome including treatment propensity correction and conditional mean centering is
\\[
Y^{DR}_i = E[Y|X_i, W=1] - E[Y|X_i, T_i=0] + \frac{T_i(Y_i-E[Y|X_i, T_i=1])}{e(X_i)} - \frac{(1-T_i)(Y_i-E[Y|X_i, T_i=0])}{1-e(X_i)}
\\]
Double robust esimation has two steps. In the first, we use effective models of our choice to estimate \\(E[Y|X_i, W=1]\\), \\(E[Y|X_i, W=0]\\) and \\(e(X_i)\\). In the second, we calculate \\(Y^{DR}_i\\) and train a model on transformed outcome variable. 

*Kang, J. D. Y., & Schafer, J. L. (2007). [Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data](https://doi.org/10.1214/07-STS227 ). Statistical Science, 22(4), 523–539.   
Knaus, M. C., Lechner, M., & Strittmatter, A. (2019). [Machine Learning Estimation of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence](https://ssrn.com/abstract=3318814). IZA Discussion Paper, 12039.*

# Non-parametric methods
Separate the individuals into groups based on their covariates and estimate the treatment effect within each group as the difference between treatment groups. Use a criterion that maximizes an approximation of the treatment effect difference between groups to separate the individuals into groups.

## Pollienated transformed-outcome tree/forest
Building a tree on the transformed outcome variable with the common CART algorithm does not return unbiased treatment estimates. Because the ratio of treatment to control group observations varies for each leaf, the average of the leaf used for prediction in CART is not a correct estimate of the treatment effect.

A better approach is to build a tree on the transformed outcome, but replace the average of the transformed outcome in each leaf \\( \bar{Y^{TO}} \\) with an estimate of the average treatment effect using the difference in the averages between treatment and control group \\(\bar{Y}(1) - \bar{Y}(0)\\). 

*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2018). [Some methods for heterogeneous treatment effect estimation in high-dimensions](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.7623). Statistics in Medicine, 37(11).*

## Causal Tree
[TODO] Explain tree building in detail

The structure of the tree is build on one random half of the training data and the leaf estimates are calculated on the other half (*honest splitting*). This avoids overfitting bias that happens when we use the data to find the most different groups and base our estimates for these groups on the same data.

*(Rzepakowsk, P., & Jaroszewics, S. (2010). [Decision Trees for Uplift Modeling](https://doi.org/10.1109/ICDM.2010.62). 2010 IEEE International Conference on Data Mining.)    
Athey, S., & Imbens, G. (2016). [Recursive partitioning for heterogeneous causal effects](https://doi.org/10.1073/pnas.1510489113). Proceedings of the National Academy of Sciences, 113(27), 7353–7360.*

## Boosted Causal Trees
*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2018). [Some methods for heterogeneous treatment effect estimation in high-dimensions](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.7623). Statistics in Medicine, 37(11).*

## Generalized Random Forest 
*Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. The Annals of Statistics, 47(2), 1148–1178.*

## Bagged Causal Multivariate Adaptive Regression Splines
Multivariate Adaptive Regression Splines (MARS) are related to the trees discussed above. 
[TODO]: I haven't seen them used in practice, but they are in The Elements of Statistical Learning and seem to do well in Powers et al.

*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2018). [Some methods for heterogeneous treatment effect estimation in high-dimensions](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.7623). Statistics in Medicine, 37(11).*

# Modified Loss Function
## Covariate Transformation
**(aka Modified Covariate Method)**    
Optimize a model \\( \tau(X_i)\\) for a loss function based 

\\[
    \underset{\tau}{\arg\min} \frac{1}{N}\sum_i (2T_i-1) \frac{T_i - e(X_i)}{4 e(X_i)(1-e(X_i))} (2(2T_i-1) Y_i - \tau(X_i))^2
\\]


*(Tian, L., Alizadeh, A. A., Gentles, A. J., & Tibshirani, R. (2014). [A simple method for estimating interactions between a treatment and a large number of covariates](https://doi.org/10.1080/01621459.2014.951443). Journal of the American Statistical Association, 109(508), 1517–1532.)    
Knaus, M. C., Lechner, M., & Strittmatter, A. (2019). [Machine Learning Estimation of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence](https://ssrn.com/abstract=3318814). IZA Discussion Paper, 12039.*


## R-learner
Optimize a model \\( \tau(X_i)\\) for a loss function based on a decomposition of the outcome function:
\\[
\underset{\tau}{\arg\min} \frac{1}{n}\sum_i \left( (Y_i − E[Y|X_i])− (T_i − E[W=1|X_i]) \tau(X_i) \right)
\\]
The nuisance function for the conditional outcome and the proponsity score are estimated separately and an second-stage model trained on the transformation loss.

The name is a hommage to Peter M. Robinson and the residualization in the decomposition. 

*Nie, X., & Wager, S. (2017). [Quasi-Oracle Estimation of Heterogeneous Treatment Effects](http://arxiv.org/abs/1712.04912). ArXiv:1712.04912.*

# Treatment Effect Projection 
Use a single model in a second stage to estimate the ITE as estimated by any method above. The second-stage model can be a linear regression for interpretability or any single model, which then replaces multiple models used in the first stage, see k-model approach. 

## X-learner 
In settings where the treatment and control group vary in size, we may want to emphasize the conditional mean model estimated on the larger group. 

Construct a treatment estimate for the treatment and control group separately using the conditional mean model from the other group: 
\\[ 
T_i^1 = Y_i(1) - E[Y(0)|X=x]
T_i^0 = E[Y(1)|X=x] - Y_i(0)
\\]

Project the treatment estimates on variables *X* directly within each group. Combine the treatment effect estimates from both projection models using a weighted average with weights manually chosen or equal to the estimated propensity score.
\\[
\hat{\tau} = w(x)\hat{\tau_0} + (1-w(x))\hat{\tau_1} 
\\]

The name refers to the *cross* use of the conditonal mean of one group in the construction of the treatment estimate for the other group.

TODO: The conditonal mean correction and propensity weighting make the X-Learner look like a variation on double robust estimation with added treatment effect projection to me.

*Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). [Metalearners for estimating heterogeneous treatment effects using machine learning](https://www.pnas.org/content/116/10/4156). Proceedings of the National Academy of Sciences, 116(10), 4156–4165.*


# Benchmark studies
These studies compare at least a subset of the methods in a structured setting:

- Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: a stepping stone toward the development of prescriptive analytics. Big Data, 6(1), 13–41. https://doi.org/10.1089/big.2017.0104
- Gubela, R. M., Bequé, A., Gebert, F., & Lessmann, S. (2019). Conversion uplift in e-commerce: A systematic benchmark of modeling strategies. International Journal of Information Technology & Decision Making, 18(3), 747–791.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.
- Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.
- Wendling, T., Jung, K., Callahan, A., Schuler, A., Shah, N. H., & Gallego, B. (2018). Comparing methods for estimation of heterogeneous treatment effects using observational data from health care databases. Statistics in Medicine, 37, 3309–3324. https://doi.org/10.1002/sim.7820