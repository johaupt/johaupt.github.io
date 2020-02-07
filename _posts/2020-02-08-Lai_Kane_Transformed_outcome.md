---
layout: post
title:  "Outcome Transformations for Uplift Modeling and Treatment Effect Estimation"
date:   2020-02-08
categories:
  - causal inference
  - uplift modeling
---


## What's the issue?

A well-known approach to estimate observation-level uplift AKA the conditional average treatment effect (CATE) is a transformation approach that turns the observed outcomes $Y_i \in \{0;1\}$ and the treatment indicator $T_i \in \{0;1\}$ into a single variable. The single variable can the be used as a target variable for any popular machine learning model. That in itself is a great idea that has been shown to work well empirically and has been discussed and refined in a number of papers. 

While the story is plausible and the empirical results are promising, the analytical discussion of the transformation approach is fragemented and, honestly, often confusing. An additional issue that the proposed transformation is only feasible for binary outcome variables, since it is explained as a recoding of Converted|Treated & Not-Converted|Not-Treated as class 1 and Non-Converted|Treated & Converted|Not-Treated as class 0. 

However, there is general approach to transform the outcome variable in the literature on causal inference. This approach seems different at first glance as we will see below. The purpose of this post is to show that the uplift transformation is in fact a rescaled special case of the more general outcome transformation approach, which works for categorical and continuous outcomes. 

## Lai's (Weighted) Outcome Transformation

The outcome transformation is introduced by Lai et al. (2006) as

$$
    Y^{Lai} = \left\{\begin{array}{lr}
        1, & \text{if } Y_i=1 \land T_i=1 \text{   or   } Y_i=0 \land T_i =0\\
        0, & \text{if } Y_i=0 \land T_i=1 \text{   or   } Y_i=1 \land T_i =0\\
        \end{array}\right.
$$

or in words, the positive cases are those that convert under treatment or haven't converted when not under treatment. 

We can summarize that as a formula

$$
     Y^{Lai} = Y_i \cdot T_i + (1-Y_i) \cdot (1-T_i)
$$

Kane et al. (2014) note that this formula ignores the ratio at which customers are treated and propose to weight the observations by the probability to fall into their observed group. The paper is confusing, but I believe what they propose can be translated to 

$$
   Y^{Kane} = Y_i \cdot \frac{T_i}{p(T)} + (1-Y_i) \cdot \frac{T_i}{1-p(T)}
$$

where $1-p(T)=p(C)$ is the probability to be in the control group. 

## The CATE-generating Outcome Transformation

The statistics literature is aware of a general outcome transformation that holds in binary and continuous cases. This transformation has apparently been known for some time, since recent papers don't bother to give a reference. 

$$
Y^{TO} = Y_i \cdot \frac{T_i}{p_i(T)} - Y_i \cdot \frac{T_i}{1-p_i(T)}
$$

The big differene to Kane et al.'s approach is that there is a $-Y_i$ instead of $1-Y_i$.

The transformed outcome (TO) has the property that its expectation is equal to the true treatment effect and is therefore used to estimate models of the conditional average treatment effect or do model selection. 
Let's confirm that the TO is equal to the true treatment effect in expectation:

\begin{align}
E[Y^{TO}] &= E[Y_i \cdot \frac{T_i}{p_i(T)} - Y_i \cdot \frac{1-T_i}{1-p_i(T)}] &&\\
&= E[Y_i] \cdot \frac{E[T_i]}{E[p_i(T)]} - E[Y_i] \cdot \frac{1-E[T_i]}{1-E[p_i(T)]}&&  |E[T_i]=p_i(T) \\
&= E[Y|T=1] - E[Y|T=0] && \textit{|Potential outcome framework assumptions} \\
&= E[Y_1 -Y_0]
\end{align}

## So what's the connection?

Let's see that the expectation of the Lai transformed outcome is. 

\begin{align}
E[Y^{Kane}] &= E[Y_i \cdot T_i + (1-Y_i) \cdot (1-T_i) &&\\
&= E[Y_i] \cdot E[T_i] + (1-E[Y_i]) \cdot (1-E[T_i])&&  |E[T_i]=p_i(T)\\
&= (1-p(T))+ E[Y|T=1] p(T) - E[Y|T=0](1-p(T)) && \textit{|Potential outcome framework assumptions} \\
\end{align}

Only in the simplest case when $p(T)=1-p(T)=0.5$ can we simplify this to

\begin{align}
E[Y^{Kane}] &= E[Y|T=1] p(T) + (1-E[Y|T=0])(1-p(T)) && |p(T)=1-p(T)=0.5 \\
&= \frac{E[Y|T=1] + (1-E[Y|T=0])}{2} &&\\
&= \frac{1+E[Y|T=1] -E[Y|T=0])}{2} &&\\
\end{align}

This is an intersting result, because papers discussing Lai's transformation have noted that $2*p(Z)-1$ gives an estimate of the true treatment effect in expectation. We can now see exactly, why this is true only when there is a 50:50 treatment control split in the data. 

Let's see that the expectation of the Kane transformed outcome is. 

\begin{align}
E[Y^{Kane}] &= E[Y_i \cdot \frac{T_i}{p(T)} + (1-Y_i) \cdot \frac{1-T_i}{1-p(T)}] &&\\
&= E[Y_i] \cdot \frac{E[T_i]}{p(T)} + (1-E[Y_i]) \cdot \frac{1-E[T_i]}{1-p(T)} &&  |E[T_i]=p(T) \\
&= 1+E[Y|T=1] - E[Y|T=0] && \textit{|Potential outcome framework assumptions} \\
&= 1+E[Y_1 -Y_0]
\end{align}

There we are! The estimates from a $Y^{Kane}$-based model are not the true treatment effect, because they are shifted upwards by a constant of 1. The constant of 1 is a direct result of using $(1-Y_i)$ in the transformation instead of $Y_i$ as proposed in the statistical literature. 

## Why does it matter?

The original Lai transformation has an interesting property. Because the transformed outcome is either 0 or 1, we can use a classifier as model to predict the treatment effect. After the correction $2*p(Z)-1$, the estimates of the treatment effect will be bounded in [-1;1] as they should be for a binary outcome variable. As we see above, however, this approach is helpful only when the treatment probability is exactly 0.5 for all observations. In other cases, we are better off using the transformed outcome and train a regression model. 

## References and additional reading

- Original proposal:    
Lai, Y.-T., Wang, K., Ling, D., Shi, H., & Zhang, J. (2006). Direct Marketing When There Are Voluntary Buyers. Proceedings of the 6th International Conference on Data Mining (ICDM), 922–927. https://doi.org/10.1109/icdm.2006.54
- Weighting for imbalanced treatment-control splits     
Kane, K., Lo, V. S. Y., & Zheng, J. (2014). Mining for the truly responsive customers and prospects using true-lift modeling: Comparison of new and existing methods. Journal of Marketing Analytics, 2(4), 218–238. https://doi.org/10.1057/jma.2014.18


- The relation of the transformed outcome to the uplift/treatment effect is somewhat discussed in     
Jaśkowski, M., & Jaroszewicz, S. (2012). Uplift Modeling for Clinical Trial data. ICML 2012 Workshop on Clinical Data Analysis. ICML 2012 Workshop on Clinical Data Analysis.    
Gutierrez, P., & Gerardy, J.-Y. (2017). Causal Inference and Uplift Modeling A review of the literature. Proceedings of the 3rd International Conference on Predictive Applications and APIs, 67, 1–13.
