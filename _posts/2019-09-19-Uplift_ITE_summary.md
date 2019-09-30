---
layout: post
title:  "Causal Machine Learning: Individualized Treatment Effects and Uplift Modeling"
date:   2019-09-19
categories:
  - causal machine learning
---

*A comprehensive collection of state-of-the-art methods from causal machine learning or uplift modeling to estimate individualized treatment effects.*

# Table of Contents
0. [K-Models Approach](#k-models-approach)
    0. [Bayesian additive regression trees](#bayesian-additive-regression-trees)
    0. [Treatment residual neural network](#treatment-residual-neural-network)
    0. [DragonNet](#dragonnet)
0. [Treatment Indicator as Variable](#treatment-indicator-as-variable)
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

Assume that we want to change the world. A little less grandiose, assume we want to take an action that will impact an outcome to be more like we prefer. A little more applied, fields like medicine and marketing make a decision to take action that increases life expectation by five years or makes grandma spend another dollar in the supermarket. To make that decision, we want to know in advance what effect the conceivable treatment will have on each individual/grandma. Remember treatment as a general term that can mean anything from a mail catalog to earlier starting time for a class to medication.     
Example questions and treatments are:

- How much more likely is a my middle-aged, sporty patient to be cured when given new medication as opposed to the standard medication?
- How much more likely are participants to complete the online training if we send them a progress report every week? Or are there certain participants that react negatively to perceived pressure?
- How much more will a customer buy if promised free shipping as opposed to not showing any banner?

Estimation of treatment effects on the individual level is a tricky problem, because we typically only have one shot at applying a treatment: either my grandma gets the coupon or she doesn't and we will be left to ponder the 'what if'. 'What if' type of questions belong to the domain of causal inference and estimating the treatment effect for individuals has become a task at the intersection of causal inference and machine learning. The same problem is known as heterogeneous treatment effects in social studies and medicine, conditional average treatment effects in econometrics and uplift modeling in business intelligence. 

The fundamental problem of 'what if' is that we can only apply one treatment to each individual and observe their outcome. Since we don't know how the individual would have reacted under alternative treatment, we can never calculate the real treatment effect. What we can do is to look at a group of individuals who have received different treatments and the difference in their outcomes.

Research into estimating treatment effects from experiments and observational studies goes far into the last century, but the last ten years have seen renewed interest to leverage machine learning techniques to increase the power of known methods or build machine learning models specialized in estimating treatment effects.    
Unfortunately, research in different fields is fractured, with different termini and notation. The following list is meant as an incomplete, but comprehensive summary of state-of-the-art methods from bioinformatics, computer science, econometrics and business studies to estimate individualized treatment effects. I will assume that we know how randomized experiments work and that we have data on some individuals who have gotten treatment and some who haven't. 

# Notation
My notation is roughly in line with the econometric literature:

Covariates for individual *i* (*What we know about the person*   ): \\( X_i \\)  
(Estimated) Propensity score (*Their chance to get the coupon from us* ): \\( P(W=w|X), e(X)=P(W=1|X) when W \in {0;1} \\)    
Treatment Group Indicator (*If the person got a coupon or not*  ): \\(W\\)    
(Potential) Outcome *Y* for individual *i* under group assignment *W* (*If/How much they bought*): \\( Y_i(W)\\)     
(Estimated) Conditional outcome under group assignment *W* (*If/How much we think they should have bought*): \\( \mu(X_i, W_i) \\)     
(Estimated) Treatment Effect (*How much impact the coupon had*): \\( \tau_i \\) 

# K-Model approach
**(aka T-Learner, Conditional Mean Regressions, Difference in Conditional Means)**    
The following approaches build a model to predict the outcome with and without treatment. By looking at the difference between the predicted outcomes, we can find out how much impact we can expect from the treatment.

When using the k-model approached, named after the number of models we need to train, we estimate an outcome model for each treatment group separately and calculate the treatment effect as the difference between the estimated outcomes.     
If there is one treatment, we will have two groups: treatment and control (where people might get nothing or Placebo). The impact of treatment vs. doing nothing is then the difference between the prediction of the model build on the treated and prediction of the model build on the control group.

The outcome models (*base learners*) can take any form, so we could use neural networks or boosted trees to do the heavy lifting. The downside of the k-model approach is that the outcome process may be difficult to estimate and that the errors of the two models in the difference may add up.

## Bayesian Additive Regression Trees
Use Bayesian Additive Regression Trees as outcome models. The difference in posterior distributions provides an uncertainty estimate of the treatment effect that may be useful for the decision or to do uncertainty based data collection. 

*Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 20(1), 217–240. https://doi.org/10.1198/jcgs.2010.08162*

## Treatment Residual Neural Network
Use neural networks as outcome models. Model calibration seems to be better if one base learner predicts the conditional mean for the control group, while the other predicts the residual between the observed outcome and the control base learner, i.e. the treatment effect.

*Farrell, M. H., Liang, T., & Misra, S. (2018). Deep Neural Networks for Estimation and Inference: Application to Causal Effects and Other Semiparametric Estimands. ArXiv E-Prints, arXiv:1809.09953*

## DragonNet
When we can't do an experiment and treatment assignment is not random, we can correct for variables that impact the treatment assignment to make unbiased estimates (*propensity weighting*). A special way to do this in a neural network is to correct the hidden layers by joint prediction of conditional means and treatment propensity in a multi-output neural network. We rely on the hidden representation to filter the information that is necessary to predict treatment assignment. 

*Shi, C., Blei, D. M., & Veitch, V. (2019). Adapting Neural Networks for the Estimation of Treatment Effects. ArXiv:1906.02120 [Cs, Stat]. Retrieved from http://arxiv.org/abs/1906.02120*


# Treatment Indicator as Variable 
**(aka S-Learner)**    

The outcome process (read: who completes their purchase in our online shop) is often more complicated than the process behind the treatment effect (read: who is impacted most by a coupon). The following approaches therefore aim to estimate the treatment effect directly without the need to build a good outcome model first.

We can include the treatment indicator as a covariate into the model, optionally with interaction to other covariates. Predict the ITE via the difference of predicting an observation with treatment set to 0 and set to 1. Training a single model for the outcome is simple and often interpretable. 

A regression model with a linear additive treatment effect and interaction effects would look like this:
\\[
 Y = \beta_0 + \tau_0 D_i+ \tau D_i X_i + \beta X_i + \epsilon_i
\\]

Under linear regression, the interaction effects between all variables and the treatment indicator blow up the dimensionality quickly. Instead, we could use any machine learning model and include the treatment indicator as a variable. However, if the treatment effect is small relative to other effects on the outcome, then regularized machine learning methods may ignore the treatment variable. 


# Outcome Transformation 
**(aka Modified Outcome Method, Class Variable Transformation, Generalized Weighted Uplift Method)**    
Our job would be so much easier if we knew the actual treatment effect and could just train a regression model to predict it, but we can never know the actual treatment effect for an individual. However, we can find an artificial variable that is equal to the treatment effect in expectation.

The proxy variable is a transformation of the observed outcome for the individual:
\\[
Y^{TO}_i = W_i Y_i(1) - (1-W_i) Y_i(0)
\\]

Or including treatment propensity correction if we didn't do a 50:50 randomized experiment:
\\[
Y^{IPW}_i = W_i \cdot \frac{Y_i(1)}{e(X_i)} - (1-W_i) \cdot \frac{Y_i(0)}{1-e(X_i)}
\\]

The transformed outcome is a noisy but unbiased estimate of the treatment effect. As an unbiased estimate, it can be used as a target variable for model training. 
The transformed outcome can also be used to calculate a feasible estimate of the MSE between model estimate and true treatment effect that is useful for model comparison. 

Hitsch, G. J., & Misra, S. (2018). Heterogeneous Treatment Effects and Optimal Targeting Policy Evaluation. SSRN.

## Double robust estimation 
The transformed outcome including treatment propensity correction and conditional mean centering is
\\[
Y^{DR}_i = E[Y|X_i, W=1] - E[Y|X_i, W_i=0] + \frac{W_i(Y_i-E[Y|X_i, W_i=1])}{e(X_i)} - \frac{(1-W_i)(Y_i-E[Y|X_i, W_i=0])}{1-e(X_i)}
\\]
Double robust esimation has two steps. In the first, we use effective models of our choice to estimate \\(E[Y|X_i, W=1]\\), \\(E[Y|X_i, W=0]\\) and \\(e(X_i)\\). In the second, we calculate \\(Y^{DR}_i\\) and train a model on transformed outcome variable. 

*Kang, J. D. Y., & Schafer, J. L. (2007). Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. Statistical Science, 22(4), 523–539. https://doi.org/10.1214/07-STS227    
Knaus, M. C., Lechner, M., & Strittmatter, A. (2019). Machine Learning Estimation of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence. IZA Discussion Paper, 12039. Retrieved from https://ssrn.com/abstract=3318814*

# Non-parametric methods
Separate the individuals into groups based on their covariates and estimate the treatment effect within each group as the difference between treatment groups. Use a criterion that maximizes an approximation of the treatment effect difference between groups to separate the individuals into groups.

## Pollienated transformed-outcome tree/forest
Build a tree on the transformed outcome, but replace the leaf estimate \\( \bar{Y}^{TO} \\) with an estimate of the average treatment effect \\(\bar{Y}(1) - \bar{Y}(0)\\). The approach is theoretically very close to causal trees, but causal trees maximize the variance between leaves for efficiency in practice. 

*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.*

## Causal Tree
*(Rzepakowsk, P., & Jaroszewics, S. (2010). Decision Trees for Uplift Modeling. https://doi.org/10.1109/ICDM.2010.62)    
Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences, 113(27), 7353–7360. https://doi.org/10.1073/pnas.1510489113*

## Boosted Causal Trees
*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.*

## Generalized Random Forest 
*Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. The Annals of Statistics, 47(2), 1148–1178.*

## Bagged Causal Multivariate Adaptive Regression Splines
Multivariate Adaptive Regression Splines (MARS) are related to the trees discussed above. 
[TODO]: I haven't seen them used in practice, but they are in The Elements of Statistical Learning and seem to do well in Powers et al.

*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.*

# Modified Loss Function
## Covariate Transformation
**(aka Modified Covariate Method)**    
Optimize a model \\( \tau(X_i)\\) for a loss function based 

\\[
    \underset{\tau}{\arg\min} \frac{1}{N}\sum_i (2W_i-1) \frac{W_i - e(X_i)}{4 e(X_i)(1-e(X_i))} (2(2W_i-1) Y_i - \tau(X_i))^2
\\]


*(Tian, L., Alizadeh, A. A., Gentles, A. J., & Tibshirani, R. (2014). A simple method for estimating interactions between a treatment and a large number of covariates. Journal of the American Statistical Association, 109(508), 1517–1532. https://doi.org/10.1080/01621459.2014.951443)    
Knaus, M. C., Lechner, M., & Strittmatter, A. (2019). Machine Learning Estimation of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence. IZA Discussion Paper, 12039. Retrieved from https://ssrn.com/abstract=3318814*


## R-learner
Optimize a model \\( \tau(X_i)\\) for a loss function based on a decomposition of the outcome function:
\\[
\underset{\tau}{\arg\min} \frac{1}{n}\sum_i \left( (Y_i − E[Y|X_i])− (W_i − E[W=1|X_i]) \tau(X_i) \right)
\\]
The nuisance function for the conditional outcome and the proponsity score are estimated separately and an second-stage model trained on the transformation loss.

The name is a hommage to Peter M. Robinson and the residualization in the decomposition. 

*Nie, X., & Wager, S. (2017). Quasi-Oracle Estimation of Heterogeneous Treatment Effects. ArXiv:1712.04912. Retrieved from http://arxiv.org/abs/1712.04912*

# Treatment Effect Projection 
Use a single model in a second stage to estimate the ITE as estimated by any method above. The second-stage model can be a linear regression for interpretability or any single model, which then replaces multiple models used in the first stage, see k-model approach. 

## X-learner 
In settings where the treatment and control group vary in size, we may want to emphasize the conditional mean model estimated on the larger group. 

Construct a treatment estimate for the treatment and control group separately using the conditional mean model from the other group: 
\\[ 
W_i^1 = Y_i(1) - E[Y(0)|X=x]
W_i^0 = E[Y(1)|X=x] - Y_i(0)
\\]

Project the treatment estimates on variables *X* directly within each group. Combine the treatment effect estimates from both projection models using a weighted average with weights manually chosen or equal to the estimated propensity score.
\\[
\hat{\tau} = w(x)\hat{\tau_0} + (1-w(x))\hat{\tau_1} 
\\]

The name refers to the *cross* use of the conditonal mean of one group in the construction of the treatment estimate for the other group.

TODO: The conditonal mean correction and propensity weighting make the X-Learner look like a variation on double robust estimation with added treatment effect projection to me.

*Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.*


# Benchmark studies
These studies compare at least a subset of the methods in a structured setting:

- Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: a stepping stone toward the development of prescriptive analytics. Big Data, 6(1), 13–41. https://doi.org/10.1089/big.2017.0104
- Gubela, R. M., Bequé, A., Gebert, F., & Lessmann, S. (2019). Conversion uplift in e-commerce: A systematic benchmark of modeling strategies. International Journal of Information Technology & Decision Making, 18(3), 747–791.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.
- Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.
- Wendling, T., Jung, K., Callahan, A., Schuler, A., Shah, N. H., & Gallego, B. (2018). Comparing methods for estimation of heterogeneous treatment effects using observational data from health care databases. Statistics in Medicine, 37, 3309–3324. https://doi.org/10.1002/sim.7820