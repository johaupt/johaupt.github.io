---
layout: post
title:  "Profitable Customer Targeting For Churn Campaigns"
date:   2020-03-16
categories:
  - econometrics
  - decision making
  - causal inference
  - churn/retention
  - direct marketing
---

*Profitable customer targeting for rentention campaigns to reduce customer churn is not simple and many of the classical approaches are flawed. This post discusses an intuitive but powerful decision rule and compares it to a popular profit function.*

A customer quitting their contract with the company or stopping to using their service  is an expensive event for any company. It is typically much more expensive to attract new customers than it to retain existing ones. The prediction of this *customer churn* and the design of retention campaigns therefore is a profitable marketing activity. 

As so often in life, when something is profitable, it is not easy and there are many ways to screw up customer targeting in retention campaigns. The consensus in the research literature is that what matters for customer targeting is the incremental effect of the campaign, often called *uplift* or *treatment effect*. An excellent way to measure incremental effects are the A/B tests that are a common tool in digital marketing by now. The next big thing to come to practice is that it is possible to predict the incremental effect on individual customers and use that to pick profitable targets.

Exactly how to make the targeting decision for a customer depends on the incremental effect of the campaign, the value of the customer and the cost of the marketing incentive. Churn campaigns typically work with a personal message that may be costly, like a phone call, and a monetary incentive to the customer, like a free add-on or a price reduction on their existing contract.

Spoiler:     
I think the best way to make a targeting decision under these circumstances is this formula

$$
p_i(1) (CLV(1) - \delta) - c > p_i(0) \cdot CLV(0)
$$

I'll explain it below, but there's a whole paper that really goes into the details [6]. 

## The profit of customer retention campaigns

The idea that the incremental effect of the campaign on each customer is a general concept that we should estimate has at least an 18 year history but has only recently really attracted attention under the umbrella of causal machine learning. The best way to think about an incremental effect is the *change* in behavior that happens *because of* the campaign. This is called a *treatment effect* in statistics and medicine because it measures the effect that some treatment has one a patient. 

That's a powerful idea that did not come naturally when people started thinking measuring the effect of retention campaigns. Many papers [1,2,3,4] on customer churn start from the profit of a churn campaign which they calculate like this [5]:

$$
\Pi = N \alpha  \left[  \beta\gamma(CLV - \delta -c) + \beta(1-\gamma)(-c) + (1-\beta)(-\delta-c)  \right] - A
$$

with    
N: Number of customers    
\(\alpha\): Ratio of customers targeted    
\(CLV\): The value of the customer to the company    
\(\beta\): Fraction of (targeted) customers who would churn    
\(\gamma\): Fraction of (targeted) customers who decide to remain when receiving the marketing incentive    
\(\delta\): The cost of the marketing incentive if it is accepted    
\(c\): The cost of contacting the customer with the marketing incentive    
\(A\): The fixed cost of running the retention campaign

There are some aspects of the formula that are relevant to calculate the overall campaign profit, but do not affect the targeting decision for a single customer, which is the purpose in all of the papers referenced above. When deciding whether to target a single customer, we can ignore the number of customers in the campaign and the fixed costs of the campaign. This leaves us:

$$
\Pi_i = \beta\gamma(CLV - \delta -c) + \beta(1-\gamma)(-c) + (1-\beta)(-\delta-c)  
$$

This formula and the story behind it have always made a lot of intuitive sense to me. In order it asks three questions: What's the outcome if the customer is a churner who can be convinced? What's the outcome if the customer is a churner but is not convinced by the offer? What's the outcome if the customer would have stayed anyway? Combine these outcomes weighted by their probability and you know how much money to expect from targeting the customer. 

Recently, I have had a hard time reconciling this perspective on customers churn campaigns with the uplift perspective on customer targeting. The uplift perspective is (or should be) based on the potential outcome framework that considers the probability of the customer to churn in the hypothetical cases of the customer receiving or not receiving the marketing incentive. The potential outcome framework is a more general framework to think about causal problems, so let's see if we can reconcile the churn campaign profit with a general perspective on decision problems and express the churn formula in the potential outcome framework.

## The targeting decision in the potential outcome framework

Our recent work [6] analyzes the customer targeting decision in the potential outcomes framework. We state the decision function as

$$
p_i(1) (CLV(1) - \delta) - c > p_i(0) \cdot CLV(0)
$$
with    
$CLV$: The value of the customer to the company    
$p_i$: The probability of a customer to stay with the company  
$\delta$: The cost of the marketing incentive if it is accepted    
$c$: The cost of contacting the customer with the marketing incentive    
$\cdot(1)$: The hypothetical value for the customer if receiving the marketing incentive    
$\cdot(0)$: The hypothetical value for the customer without the marketing incentive

We'll follow the implicit assumption in the campaign profit function that the value of the customer is not influenced by the marketing incentive.

$$
p_i(1) (CLV - \delta) - c > p_i(0) \cdot CLV
$$

which we rearrange into 

$$
(p_i(1) - p_i(0))CLV - p_i(1) c_1 - c > 0
$$

The formula tells us that we should target a customer if the expected profit is higher when the customer is treated (left) than when the customer is not treated (right). The profit if the customer gets no incentive is the chance that they will stay without treatment times their value. The profit if the customer gets the incentive is the chance that they will stay after the incentive times the value minus the incentive. And if we send the incentive, we have to pay for contacting the customer.

## Connecting campaign profit and the potential outcome framework

Let's see if we can bring the two together. We'll start with the expanded version of the churn campaign profit:

$$
\Pi_i = \beta \gamma(CLV - \delta -c) + \beta(1-\gamma)(-c) + (1-\beta)\gamma(-\delta-c)
$$

Let's multiply the whole thing out and rearrange the pieces.

$$
\Pi_i = \beta \gamma(CLV) + \beta \gamma (\color{blue}{-\delta}) + \beta \gamma (\color{red}{-c}) + \beta(1-\gamma)(\color{red}{-c}) + (1-\beta)(\color{blue}{-\delta}) +(1-\beta)(\color{red}{-c})
$$

I've color-coded the cost parameters, because we will try to summarize them by using the fact that $\beta$ and $(1-\beta)$ and $\gamma$ and $(1-\gamma)$ respectively are probabilities that add up to 1.

\begin{align}
\Pi_i &= \beta \gamma(CLV) + 
\beta\gamma (\color{blue}{-\delta}) + (1-\beta) (\color{blue}{-\delta}) + \beta (\color{red}{-c}) + (1-\beta)(\color{red}{-c})  \\
       &= \beta \gamma(CLV) + \beta\gamma \color{blue}{-\delta} + (1-\beta)\color{blue}{-\delta} - \color{red}{c}\\
       &= \beta \gamma(CLV) - \color{blue}{\delta}(\beta\gamma + 1 -\beta) - \color{red}{c}\\
       &= \beta \gamma(CLV) - \color{blue}{\delta}(1-\beta(1-\gamma)) - \color{red}{c}
\end{align}

We will target a customer if the profit is positive, i.e.

$$
\beta \gamma CLV - (1-\beta(1-\gamma)) \delta - c > 0
$$

This looks like the decision under the potential outcome framework if    

\begin{align}
p_i(1)-p_i(0) &= \beta\gamma   \\         
p_i(1) &= (1-\beta(1-\gamma)) \\
\text{and following from these}    \\
p(0) &= 1-\beta
\end{align}

Does that make sense? Let's see:    

1. $p(0)$ is the probability of a customer to make the plan to churn. $\checkmark$    
2. $p(1)$ is the complementary probablility of a customer to make a plan to churn and churn even when offered the treatment. That includes customers who make no plan to churn and those that are convinced by the incentive. $\checkmark$    
3. $p(1)-p(0)$ is the probability of a customer to make the plan to churn and not churn when offered the treatment. **?**

Interpretation 3 (refering to Eq.1) is weird because the treatment effect $p(1)-p(0)$ is in principle bounded between [-1,1]. It is reasonable in churn campaigns that some customers will be reminded by the campaign to cancel their contract and we see that quite severely in real campaigns. 

To find a solution and correctly map all potential treatment effects, let's consider the extreme cases in the current profit formula:

1. $\beta\gamma=1$ if $\beta=\gamma=1$. In words, the maximum effectiveness of a campaign is reached when all customers consider to churn before the campaign and all will accept the incentive to stay. 
2. $\beta\gamma=0$ if either $\beta$ or $\gamma$ or both are zero. In words, the campaign has no effect if no customers consider to churn or no customers accept the marketing incentive when offered. 

The problem is apparent in point 2. When no customers consider churning, then the campaign may still have a negative effect that leads some customer to churn. We will have to alter the profit function to allow for a negative effect.

## Campaign profit as a targeting decision

Note that the probabilites $\beta$ and $\gamma$ define the set of states that are possible after targeting a customer and assign a value to each state. $\beta$ indicates if a customer has plans to leave the customer and $\gamma$ indicates if she will stay after receiving the incentive. 

The churn literature usually assumes that $\lambda=1$ if the customer had no plans to churn, because it seems plausible that everybody would accept an offer that aligns with their original plan. Let's remove that assumption for now:

$$
\Pi_i = \color{red}{\beta \gamma}(CLV - \delta -c) + \color{blue}{\beta(1-\gamma)}(-c) + \color{salmon}{(1-\beta)\gamma}(-\delta-c)  + \color{purple}{(1-\beta)(1-\gamma)}(?)
$$

We can construct a state-payoff matrix from the formula that makes the underlying logic even clearer. The columns indicate if a customer has plans to churn before receiving the incentive, the rows indicate if they churn after receiving the incentive. 

||  1 | 0 |
|------------------|------------------ |-------------------------|
|**1**| $\color{red}{CLV - \delta -c}$    | $\color{salmon}{- \delta -c}$  |
|**0**| $\color{blue}{-c}$   | $\color{purple}{?}$  |

The state-payoff matrix suggested by the profit function is useful because we can compare it to the state-payoff matrix that we would construct from the company cash flow. If we offer an incentive to a customer, we will always pay the offer cost $c$. If the customer accepts the incentive and stays on, we will additionally have to make good on the promise of the incentive and will incur an additional cost $\delta$ in the first row. Whenever the customer stays, we will receive their spending in form of the customer lifetime value.

||  1 | 0 |
|------------------|------------------ |-------------------------|
|**1**| $CLV - \delta -c$    | $CLV - \delta -c$  |
|**0**| $-c$   | $CLV-c$  |

The cash-flow includes an additional CLV in the second column, but column-wise linear transformation will not change the decision. Substracting CLV in the second column returns the state-payoff matrix from the profit function above and allows us to fill in the missing payoff if a customer had no plans to churn but does not accept the offer.

||  1 | 0 |
|------------------|------------------ |-------------------------|
|**1**| $\color{red}{CLV - \delta -c}$    | $\color{salmon}{- \delta -c}$  |
|**0**| $\color{blue}{-c}$   | $\color{purple}{-c}$  |

With this state-payoff matrix, let's come back to the possibility of negative reactions to the incentive. We have seen that we unreasonably restrict the actions available to the customer. A customer can accept the offer, they can reject the offer *and they can be triggered by the offer to leave*. I suggest that a good way to model this is to consider a third case, where the customer reacts negatively to the treatment. If they had the plan to churn anyway, then this will not change the outcome. If they were considering to stay, then this will decrease the profit of the campaign by removing their CLV.

||  1 | 0 |
|------------------|------------------ |-------------------------|
|**1**| $\color{red}{CLV - \delta -c}$    | $\color{salmon}{- \delta -c}$  |
|**0**| $\color{blue}{-c}$   | $\color{purple}{-c}$  |
|**-1**| $-c$   | $-CLV-c$  |

Let's denote the probability that a customer will react negatively to the offer by $\lambda$. Then there is the probability that a customer will accept the offer $\gamma$ and a remaining option to not accept the offer with a probability $1-\gamma-\lambda$.

I'll follow the reasoning of the literature and assume that all customer who have no plans to churn will not reject the offer, i.e. that $(1-\beta)(1-\gamma-\lambda)=0$. That assumption is not strictly necessary for the analysis, but allows us to stick close to the original profit formula. 

The the enhanced churn campaign profit formula is (with changes in green)

$$
\Pi_i = \beta\gamma(CLV - \delta -c) + \color{green}{(1-\beta)\lambda(-CLV-c)} + \beta(1-\gamma\color{green}{-\lambda})(-c) + (1-\beta)(-\delta-c)  
$$

Following the same math as before, this looks like the decision under the potential outcome framework if    

\begin{align}
p_i(1)-p_i(0) &= \beta\gamma - (1-\beta)\lambda   \\         
p_i(1) &= (1-\beta(1-\gamma)) \\
\text{and following from these}    \\
p(0) &= 1-\beta
\end{align}

Does that make sense? Let's see:    

1. $p(0)$ is the probability of a customer to make the plan to churn. $\checkmark$    
2. $p(1)$ is the complementary probablility of a customer to make a plan to churn and churn even when offered the treatment. That includes customers who make no plan to churn and those that are convinced by the incentive. $\checkmark$    
3. $p(1)-p(0)$ is the probability of a customer to make the plan to churn and not churn when offered the treatment **minus the probability of the customer to have no plan to churn but to churn when offered the treatment**.$\checkmark$
(If you are familiar with the transformed outcome approach, compare that explanation to how the transformed outcome is calculated.)

Let's again consider the extreme cases in the enhanced profit formula:

1. $\beta\gamma - (1-\beta)\lambda=1$ if $\beta=\gamma=1$. In words, the maximum effectiveness of a campaign is reached when all customers consider to churn before the campaign and all will accept the incentive to stay. 
2. $\beta\gamma - (1-\beta)\lambda=0$ if A) $\beta=1$ and $\gamma=0$, B) $\beta=0$ and $\lambda=0$ or C) $\beta\gamma = (1-\beta)\lambda$. In words, the campaign has no effect if A) all customer consider to churn but nobody reacts to the incentive, B) no customers consider to churn and the campaign has no negative effects or C) if the number of customer who consider to churn and are convinced by the campaign and the number of customer who were not planning to churn but are negatively affected by the campaign is exactly equal. 
3. $\beta\gamma - (1-\beta)\lambda=-1$ if $(1-\beta)=\lambda=1$ i.e. $\beta=0$ and $\lambda=1$. In words, the campaign has the maximum negative effect if no customer were planning to churn, but all will churn after receiving the incentive. 

## Conclusion

It may be reasonable in some applications to assume that there are no adversarial campaign effects, $\lambda=0$, but churn campaigns are not one of them. We propose a more intuitive (and correct) way to make the targeting decision:

$$
p_i(1) (CLV(1) - \delta) - c > p_i(0) \cdot CLV(0)
$$

If you want to apply the campaign churn profit function that is popular in the literature, you should use the corrected function that allows for negative effects:

$$
\Pi = N \alpha  \left[ \beta\gamma(CLV - \delta -c) + \color{green}{(1-\beta)\lambda(-CLV-c)} + \beta(1-\gamma\color{green}{-\lambda})(-c) + (1-\beta)(-\delta-c) \right] -A
$$

where $\lambda$ is the probability that a customer will leave after receiving the marketing incentive.

## References

[1] Lessmann, S., Haupt, J., Coussement, K., & De Bock, K. W. (2019). Targeting customers for profit: An ensemble learning framework to support marketing decision making. Information Sciences, In Press. https://doi.org/10.1016/j.ins.2019.05.027

[2] Óskarsdóttir, M., Baesens, B., & Vanthienen, J. (2018). Profit-based model selection for customer retention using individual customer lifetime values. Big Data, 6(1), 53–65. https://doi.org/10.1089/big.2018.0015

[3] Lemmens, A., & Gupta, S. (2017). Managing Churn to Maximize Profits (SSRN Scholarly Paper ID 2964906; SSRN Working Paper Series). Social Science Research Network. https://dx.doi.org/10.2139/ssrn.2964906

[4] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. European Journal of Operational Research, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031

[5] Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Mason, C. H. (2006). Defection detection: Measuring and understanding the predictive accuracy of customer churn models. Journal of Marketing Research, 43(2), 204–211. https://doi.org/10.1509/jmkr.43.2.204

[6] Haupt, J. & Lessmann, S. (2020). Targeting Customers under Response-Dependent Costs. https://arxiv.org/abs/2003.06271
