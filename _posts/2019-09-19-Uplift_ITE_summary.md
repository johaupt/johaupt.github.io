---
layout: post
title:  "Causal Machine Learning: Individualized Treatment Effects and Uplift Modeling"
date:   2019-09-19
categories:
  - causal machine learning
---

*A comprehensive collection of methods to estimate individualized treatment effects with a focus on machine learning, known as causal machine learning or uplift modeling.*

# Table of Contents
0. [Linear Additive Treatment Variable](#linear-additive-treatment-variable)
0. [Outcome Transformation](#outcome-transformation)
    0. [Double robust estimation](#double-robust-estimation)
0. Modified Loss Function
    0. [Modified Covariate Method](#modified-covariate-method)
    0. [R-learner](#r-learner)
    0. [Pollienated transformed-outcome tree](#pollienated-transformed-outcome-tree)
0. [Causal Tree](#causal-tree)
    0. [Boosted causal trees](#boosted-causal-trees)
    0. [Generalized random forest](#generalized-random-forest)
0. [Bagged Causal MARS](#bagged-causal-mars)
0. [K-Models Approach](#k-models-approach)
    0. [Bayesian additive regression trees](#bayesian-additive-regression-trees)
    0. [Treatment residual neural network](#treatment-residual-neural-network)
0. [DragonNet](#dragonnet)
0. [Estimated Treatment Effect Projection](#treatment-effect-projection)
    0. [X-learner](#x-learner)

The estimation of treatment effects on the individual level has become a task at the intersection of causal inference and machine learning. The same problem is known as heterogeneous treatment effects in social studies and medicine, conditional average treatment effects in econometrics and uplift modeling in information systems. 

The idea is generally to estimate the effect that a conceivable treatment will have on an individual. Treatment is a general term that can mean anything from a mail catalog to earlier starting time for class to medication. Example questions and treatments are:
- How much more likely is a my middle-aged, sporty patient to be cured when given new medication as opposed to the standard medication?
- How much more likely are participants to complete the online training if we send them a progress report every week? Or are there certain participants that react negatively to perceived pressure?
- How much more will a customer buy if promised free shipping as opposed to not showing any banner?

The problem is equally general in that we can only apply one treatment to each individual and observe their outcome. Since we don't know how the individual would have reacted under alternative treatment, we can never calculate the real treatment effect. What we can do is to look at a group of individuals who have received different treatments and the difference in their outcomes.  

Research into estimating treatment effects from observed outcomes goes far into the last century, but the last ten years have seen renewed interest to leverage machine learning techniques to increase the power of known methods or build machine learning models specialized in estimating treatment effects.

Unfortunately, research in different fields is fractured, with different termini and notation. The following list is meant as an incomplete, but comprehensive collection of methods from bioinformatics, computer science, econometrics and business studies to estimate individualized treatment effects.

# Notation
Covariates for individual *i*: \\( X_i \\)   
Treatment Group Indicator: \\(W\\)   
(Potential) Outcome *Y* for individual *i* under group assignment *W*: \\( Y_i(W)\\)   
(Estimated) Propensity score \\( e(X) = P(W|X) \\)    
(Estimated) Conditional outcome under group assignment *G*: \\( \mu(X_i, W_i) \\)

# Benchmark studies
These studies compare at least a subset of the methods in a structured setting:

- Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: a stepping stone toward the development of prescriptive analytics. Big Data, 6(1), 13–41. https://doi.org/10.1089/big.2017.0104
- Gubela, R. M., Bequé, A., Gebert, F., & Lessmann, S. (2019). Conversion uplift in e-commerce: A systematic benchmark of modeling strategies. International Journal of Information Technology & Decision Making, 18(3), 747–791.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.
- Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.
- Wendling, T., Jung, K., Callahan, A., Schuler, A., Shah, N. H., & Gallego, B. (2018). Comparing methods for estimation of heterogeneous treatment effects using observational data from health care databases. Statistics in Medicine, 37, 3309–3324. https://doi.org/10.1002/sim.7820



The following approaches can be classified as *direct methods*:    

# Linear Additive Treatment Variable 
**(S-Learner)**    
Include treatment indicator as a covariate into the model, optionally with interaction to other covariates. Predict the ITE via the difference of predicting an observation with treatment set to 0 and set to 1.

Advantages:
- Single model
- Interpretable

Disadvantages: 
- Treatment effect typically small relative to other effects
- Model might ignore treatment variable


# Outcome Transformation 
**(Modified Outcome Method, Class Variable Transformation, Generalized Weighted Uplift Method)** 
The transformed outcome is a noisy but unbiased estimate of the treatment effect and can as such be used as a target variable for model training. It can be used to calculate a feasible estimate of the MSE between model estimate and true treatment effect that is useful for model comparison. 

The transformed outcome is:

\\[
Y^*_i = W_i Y_i(1) - (1-W_i) Y_i(0)
\\]

The transformed outcome including treatment propensity correction is:

\\[
Y^*_i = W_i \cdot \frac{Y_i(1)}{e(X_i)} - (1-W_i) \cdot \frac{Y_i(0)}{1-e(X_i)}
\\]

## Double robust estimation 
The transformed outcome including treatment propensity correction and conditional mean centering is:
TODO

# Modified Loss Function
## Modified Covariate Method 
**(Covariate Transformation)**
Optimize a model \\( \tau(X_i)\\) for a loss function based 

\\[
    \underset{\tau}{\arg\min} \frac{1}{N}\sum_i (2W_i-1) \frac{W_i - p(X)}{4p(X_i)(1-p(X))} (2(2W_i-1) Y_i - \tau(X_i))^2
\\]


*(Tian, L., Alizadeh, A. A., Gentles, A. J., & Tibshirani, R. (2014). A simple method for estimating interactions between a treatment and a large number of covariates. Journal of the American Statistical Association, 109(508), 1517–1532. https://doi.org/10.1080/01621459.2014.951443)    
Knaus, M. C., Lechner, M., & Strittmatter, A. (2019). Machine Learning Estimation of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence. IZA Discussion Paper, 12039. Retrieved from https://ssrn.com/abstract=3318814*


## R-learner
Optimize a model \\( \tau(X_i)\\) for a loss function based on a decomposition of the outcome function:
\\[
\underset{\tau}{\arg\min} \frac{1}{n}\sum_i \left( (Y_i − E[Y|X])− (W_i − E[W=1|X_i]) \tau(X_i) \right)
\\]
The nuisance function for the conditional outcome and the proponsity score are estimated separately and an second-stage model trained on the transformation loss.

The name is a hommage to Peter M. Robinson and the residualization in the decomposition. 

*Nie, X., & Wager, S. (2017). Quasi-Oracle Estimation of Heterogeneous Treatment Effects. ArXiv:1712.04912. Retrieved from http://arxiv.org/abs/1712.04912*


## Pollienated transformed-outcome tree/forest
Build trees on the transformed outcome, but replace the leaf estimates with \\(\bar{Y}(1) - \bar{Y}(0)\\). The approach is theoretically very close to causal trees, but causal trees maximize the variance between leaves for efficiency in practice. 

*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.*


# Causal Tree
Build a tree with a splitting criterion that maximizes an estimate of the treatment effect between groups. 

*(Rzepakowsk, P., & Jaroszewics, S. (2010). Decision Trees for Uplift Modeling. https://doi.org/10.1109/ICDM.2010.62)    
Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences, 113(27), 7353–7360. https://doi.org/10.1073/pnas.1510489113*

## Boosted Causal Trees
*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.*

## Generalized Random Forest 
*Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. The Annals of Statistics, 47(2), 1148–1178.*

# Bagged Causal MARS
*Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.*

The following approaches can be classified as *Indirect Models, Multi-model Approaches or Metalearners*:    

## K-Model approach
**(T-Learner, Conditional Mean Regressions, Difference in Conditional Means)**    
Estimate an outcome model for each treatment group separately and calculate the treatment effect as the difference between the estimated outcomes.
The outcome models (*base learners*) can take any form.

### Bayesian Additive Regression Trees
Use Bayesian Additive Regression Trees as a base learner. The difference in posterior distributions provides an uncertainty estimate of the treatment effect.

*Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 20(1), 217–240. https://doi.org/10.1198/jcgs.2010.08162*

### Treatment Residual Neural Network
Use Neural Networks as base learner. Model calibration seems to be better if one base learner predicts the conditional mean for the control group, while the other predicts the residual between the observed outcome and the control base learner, i.e. the treatment effect.

*Farrell, M. H., Liang, T., & Misra, S. (2018). Deep Neural Networks for Estimation and Inference: Application to Causal Effects and Other Semiparametric Estimands. ArXiv E-Prints, arXiv:1809.09953*

### DragonNet
Under non-random treatment assignmnet, it is sufficient to correct for variables that impact the treatment assignment. Thus we correct for non-random treatment assignment by joint prediction of conditional means and treatment propensity in a multi-output neural network. We rely on the hidden representation to filter the information that is necessary to predict treatment assignment. 

*Shi, C., Blei, D. M., & Veitch, V. (2019). Adapting Neural Networks for the Estimation of Treatment Effects. ArXiv:1906.02120 [Cs, Stat]. Retrieved from http://arxiv.org/abs/1906.02120*


# Treatment Effect Projection 
Use a single model in a second stage to estimate the ITE as estimated by any method above. The second-stage model can be a linear regression for interpretability or any single model, which then replaces multiple models used in the first stage, see k-model approach. 

## X-learner 
In settings where the treatment and control group vary in size, we may want to emphasize the conditional mean model estimated on the larger group. 

Construct a treatment estimate for the treatment and control group separately using the conditional mean model from the other group: 
\\[ 
D_i^1 = Y_i(1) - E[Y(0)|X=x]
D_i^0 = E[Y(1)|X=x] - Y_i(0)
\\]

Project the treatment estimates on variables *X* directly within each group. Combine the treatment effect estimates from both projection models using a weighted average with weights, for example, equal to the estimated propensity score.
\\[
\hat{\tau} = w(x)\hat{\tau_0} + (1-w(x))\hat{\tau_1} 
\\]

The name refers to the *cross* use of the conditonal mean of one group in the construction of the treatment estimate for the other group.

TODO: The conditonal mean correction and propensity weighting make the X-Learner look like a variation on double robust estimation to me. Verify!

*Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.*


