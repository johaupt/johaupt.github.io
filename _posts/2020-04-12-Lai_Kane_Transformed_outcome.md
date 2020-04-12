---
layout: post
title:  "Outcome Transformations for Uplift Modeling and Treatment Effect Estimation"
date:   2020-04-12
categories:
  - causal inference
  - uplift modeling
  - outcome transformation
---



## What's the issue?

The outcome variable transformation approach to estimate observation-level uplift (a.k.a. the conditional average treatment effect CATE) is a transformation that turns the observed outcomes \\(Y_i \in \{0;1\}\\) and the treatment indicator \\(T_i \in \{0;1\}\\) into a single outcome. This single variable can the be used as a target variable for any popular machine learning model. That in itself is a great idea that has been shown to work well empirically and has been discussed and refined in a number of papers. 

While the story is plausible and the empirical results are promising, the analytical discussion of the transformation approach is fragemented and, honestly, often confusing. An additional issue that the proposed transformation is only feasible for binary outcome variables, since it is explained as a recoding of profitable customers, (Converted\\(\land\\)Treated) \\(\lor\\) (Not-Converted\\(\land\\)Not-Treated), as class 1 and unprofitable customers, (Non-Converted\\(\land\\)Treated) \\(\lor\\) (Converted\\(\land\\)Not-Treated), as class 0. 

However, there is general approach to transform the outcome variable in the literature on causal inference. This approach seems different at first glance as we will see below. The purpose of this post is to show that the uplift transformation is in fact a rescaled special case of the more general outcome transformation approach, for which we have a solid analytical foundations and which works for categorical and continuous outcomes. 

## Lai's (Weighted) Outcome Transformation

The outcome transformation is introduced by Lai et al. (2006) as

$$
    Y^{Lai} = \left\{\begin{array}{lr}
        1, & \text{if } Y_i=1 \land T_i=1 \text{   or   } Y_i=0 \land T_i =0\\
        0, & \text{if } Y_i=0 \land T_i=1 \text{   or   } Y_i=1 \land T_i =0\\
        \end{array}\right.
$$

or in words, the positive cases for targeting are those that convert under treatment and don't convert when not under treatment. 

We can summarize that as a formula

$$
     Y^{Lai} = Y_i \cdot T_i + (1-Y_i) \cdot (1-T_i)
$$

Kane et al. (2014) note that this formula ignores the ratio at which customers are treated. They propose to weight the observations by the probability to fall into their observed group, but I find the paper confusing. I believe what they propose can be translated to 

$$
   Y^{Kane} = Y_i \cdot \frac{T_i}{p_T} + (1-Y_i) \cdot \frac{1-T_i}{1-p_T}
$$

where \\( 1-p_T=p_C \\) is the probability to be in the control group. The probability to receive treatment may dependent on characteristics \\( X \\) as in \\( p_T=p(T=1 \| X) \\).

## The CATE-generating Outcome Transformation

The statistics literature has long been aware of a general outcome transformation that holds in binary and continuous cases. Wooldridge's (2001) book on panel data analysis has a decent treatment on propensity weighted estimation and its background starting in the 50s.

$$
\begin{align*}
Y^{TO} &= Y_i \cdot \frac{T_i}{p_T} - Y_i \cdot \frac{1-T_i}{1-p_T}
\end{align*}
$$

The big differene to Kane et al.'s approach is that the outcome can be a categorical or continuous variables which is transformed using \\(-Y_i\\) instead of \\(+(1-Y_i)\\). I find that a good way to think about the outcome transformation is that each converter in the treatment group *increases* the overall treatment effect, while each converter in the control group *reduces* the potential effect of the treatment. The latter is less obvious, but we can image a case where every individual in the control group converts naturally and the treatment effect is zero because no increase on conversion is possible.

The transformed outcome (TO) has the property that its expectation is equal to the true treatment effect and is therefore used to estimate models of the conditional average treatment effect or do model selection. Wooldridge's (2010) Econometric Analysis of Cross Section and Panel Data discusses steps of the proof. It's convenient to use an equivalent formulation of the TO for the proof and note that I'll drop the index \\(i\\) from here.

$$
\begin{align*}
Y^{TO} &= Y \cdot \frac{T}{p_T} - Y \cdot \frac{1-T}{1-p_T} \\
       &= Y \cdot \frac{T(1-p_T)}{p_T(1-p_T)} - Y \cdot \frac{(1-T)p_T}{(1-p_T)p_T}\\
       &= Y \cdot \left(\frac{T(1-p_T)-(1-T)p_T}{p(1-p_T)}\right)\\
       &= Y \frac{T-p_T}{p_T(1-p_T)} \\
       &= [T Y_1 + (1-T) Y_0] \frac{T-p_T}{p_T(1-p_T)}
\end{align*}
$$

In the last step, we've rewritten the vector of observed outcomes \\(Y\\) as a sum of the potential outcomes \\(Y_0, Y_1\\) as \\(Y=T Y_1 + (1-T) Y_0\\).


Let's confirm that the TO is equal to the true treatment effect in expectation conditional on \\(X\\):

$$
\begin{align*}
E[Y^{TO}|X] &= E [ (T Y1 + (1-T) Y(0)) \frac{T-p_T}{p_T(1-p_T)} |X ]  \\
&= \frac{1}{{p_T(1-p_T)}} E [ T Y_1 (T-p_T) + (1-T) Y_0 (T-p_T) |X ]  \\
&= \frac{1}{{p_T(1-p_T)}} E [ Y_1 T (1-p_T) +  Y_0 p_T (1-T) |X ]  && | T^2=T \\
&= \frac{1}{{p_T(1-p_T)}} E [ Y_1 T (1-p_T) +  Y_0 p_T - T Y_0 p_T |X ] \\
&= E[Y_1|X] - E[Y_0|X] && | Overlap \\
&= E[Y_1 -Y_0|X]
\end{align*}
$$

.
## So what's the connection?

Let's see that the expectation of the Lai transformed outcome is. 

$$
\begin{align*}
E[Y^{Lai}|X] &= E[Y \cdot T + (1-Y) \cdot (1-T)|X] && | Y = TY_1 + (1-T)Y_0 \\
&= E[Y_1|X] \cdot E[T|X] + (1-E[Y_0|X]) \cdot (1-E[T|X]) && \textit{|Unconfoundedness}\\
&= (1-p_T)+ E[Y_1|X] p_T - E[Y_0|X](1-p_T) &&  |E[T|X]=p_T \\
\end{align*}
$$

Only in the simplest case when \\(p_T=1-p_T=0.5\\) can we further simplify this to

$$
\begin{align}
 &= E[Y_1|X] p_T + (1-E[Y_0|X])(1-p_T) && |p_T=1-p_T=0.5 \\
&= \frac{E[Y_1|X] + (1-E[Y_0|X])}{2} &&\\
&= \frac{1+E[Y_1|X] -E[Y_0|X])}{2} &&\\
\end{align}
$$

This is an intersting result, because papers discussing Lai's transformation have noted that \\(2 \cdot p(Y^{Lai}=1)-1\\) gives an estimate of the true treatment effect in expectation. We can now see exactly why this holds only when there is a 50:50 treatment control split in the data. 

Let's check the expectation of the Kane transformed outcome with an arbitrary probability of treatment:

$$
\begin{align}
E[Y^{Kane}|X] &= E[Y \cdot \frac{T}{p_T} + (1-Y) \cdot \frac{1-T}{1-p_T}|X] && \textit{|Unconfoundedness} \\
&= E[Y_1|X] \cdot \frac{E[T|X]}{p(T)} + (1-E[Y_0|X]) \cdot \frac{1-E[T|X]}{1-p(T)} &&  |E[T|X]=p_T \\
&= 1+E[Y_1|X] - E[Y_0|X] 
\end{align}
$$

There we are! The estimates from a \\(Y^{Kane}\\)-based model are not the true treatment effect, because they are shifted upwards by a constant of 1. The constant of 1 is a direct result of using \\((1-Y)\\) for outcomes in the control group instead of \\(-Y\\) as proposed in the statistical literature. 

## Why does it matter?

The original Lai transformation has an interesting property. Because the transformed outcome is either 0 or 1, we can use a classifier as model to predict the treatment effect. After the correction \\(2 \cdot p(Y^{Lai}=1)-1\\), the estimates of the treatment effect will be bounded in [-1;1] as they should be for a binary outcome variable. As we see above, however, this approach is helpful only when the treatment probability is exactly 0.5 for all observations. 

In other cases, we are better off using the transformed outcome and train a regression model. The prediction of the regression model is not naturally bounded in [-1;1]. In practice, clipping the prediction to values within reasonable bounds may help to avoid implausible predictions, despite not being very elegant. 

## References and additional reading

- Original proposal:    
Lai, Y.-T., Wang, K., Ling, D., Shi, H., & Zhang, J. (2006). Direct Marketing When There Are Voluntary Buyers. Proceedings of the 6th International Conference on Data Mining (ICDM), 922–927. https://doi.org/10.1109/icdm.2006.54

- Weighting for imbalanced treatment-control splits     
Kane, K., Lo, V. S. Y., & Zheng, J. (2014). Mining for the truly responsive customers and prospects using true-lift modeling: Comparison of new and existing methods. Journal of Marketing Analytics, 2(4), 218–238. https://doi.org/10.1057/jma.2014.18


- The relation of the transformed outcome to the uplift/treatment effect is somewhat discussed in     
Jaśkowski, M., & Jaroszewicz, S. (2012). Uplift Modeling for Clinical Trial data. ICML 2012 Workshop on Clinical Data Analysis. ICML 2012 Workshop on Clinical Data Analysis.    
Gutierrez, P., & Gerardy, J.-Y. (2017). Causal Inference and Uplift Modeling A review of the literature. Proceedings of the 3rd International Conference on Predictive Applications and APIs, 67, 1–13.
