---
title: "RL — Proximal Policy Optimization (PPO) Explained"
source: "https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12"
author:
  - "[[Jonathan Hui]]"
published: 2018-09-17
created: 2026-09-08
description: "RL — Proximal Policy Optimization (PPO) Explained A quote from OpenAI on PPO: Proximal Policy Optimization (PPO), which perform comparably or better than state-of-the-art approaches while being …"
tags:
  - "clippings"
---
![](https://miro.medium.com/v2/resize:fit:2000/format:webp/0*wNFAvzs5NPtr--d1)

Photo by Pietro De Grandi

A quote from OpenAI on PPO:

> Proximal Policy Optimization (PPO), which perform comparably or better than state-of-the-art approaches while being much simpler to implement and tune.

Actually, this is a very humble statement comparing with its real impact. Policy Gradient methods have convergence problem which is addressed by the natural policy gradient. However, in practice, natural policy gradient involves a second-order derivative matrix which makes it not scalable for large scale problems. The computational complexity is too high for real tasks. Intensive research is done to reduce the complexity by approximate the second-order method. PPO uses a slightly different approach. Instead of imposing a hard constraint, it formalizes the constraint as a penalty in the objective function. By not avoiding the constraint at all cost, we can use a first-order optimizer like the Gradient Descent method to optimize the objective. Even we may violate the constraint once a while, the damage is far less and the computation is much simple. Let’s go through quickly on the basic concepts before explaining PPO in details.

## Minorize-Maximization MM algorithm

How can we optimize a policy to maximize the rewards?

With the Minorize-Maximization **MM** algorithm, this is achieved **iteratively** by maximizing a lower bound function ***M*** (the blue line below) approximating the expected reward *η* locally*.*

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JR_HgKDpNIzOX1t6JE7AQA.jpeg)

Modified from source

First, we start with an initial policy guess and find a lower bound *M* for *η* at this policy. We optimize M and use the optimal policy for *M* as the next guess. We approximate a new lower bound again at the new guess and repeat the iterations until the policy converges. To make it works, we do need to find a lower bound *M* that is easier to optimize.

## Line search

There are two major optimization methods: the line search like the gradient descent and the trust region. Gradient descent is easy, fast and simple in optimizing an objective function. This is why it is so popular in deep learning even more accurate methods are available.

Line search first picks the steepest direction and then move forward by a step size. But how can this strategy go wrong in reinforcement learning RL? Let’s take a robot to the Angels Landing for hiking. As shown below, we ascend the hill by determining the direction first. If the step size is too small, it will take forever to get to the peak. But if it is too big, we can fall down the cliff. Even if the robot survives the fall, it lands in areas with height much lower than where we were. Policy gradient is mainly an on-policy method. It searches actions from the current state. Hence, we resume the exploration from a bad state with a locally bad policy. This hurts performance badly.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*k9NFvgjf2XGiLiiarp0Q2Q.jpeg)

Modified from Source

## Trust region

In the trust region, we determine the maximum step size that we want to explore first (the yellow circle below). Then we locate the optimal point within the trust region and resume the search from there.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*eQDsFaMkSXw0g6gVIErXaQ.jpeg)

Modified from Source

**What is the maximum step size in a trust region?** In the trust region method, we start with an initial guess. Optionally, we can readjust the region size dynamically. For example, we can shrink the region if the divergence of the new and current policy is getting large (or vice versa). In order not to make bad decisions, we can shrink the trust region if the policy is changing too much.

In PPO, we limit how far we can change our policy in each iteration through the KL-divergence. KL-divergence measures the difference between two data distributions *p* and *q*.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*N1BvIhBD9vP5vlY-3eppLQ.png)

We repurpose it to measure the difference between the two policies. We don’t want any new policy to be too different from the current one.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3QNN7laZTbW7guFT-YnJNg.png)

Source: Wikipedia

So how can we limit policy change to make sure we don’t make bad decisions? It turns out we can find a lower bound function ***M*** as

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Zx6Wd8SurELVlsmfF51wcA.png)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*FQDiuNJ1WptAN3LnBKYzWw.jpeg)

with ***L(θ)*** equals

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*QZEVk1dguz_7dXO1TvYUQA.jpeg)

and the second terms is the KL-divergence.

*L* is the expected advantage function (the expected rewards minus a baseline like V(s)) for the new policy. It is estimated by an old (or current) policy and then recalibrate using the probability ratio between the new and the old policy. ==We use the advantage function instead of the expected reward because it reduces the variance of the estimation==. As long as the baseline does not dependent on our policy parameters, the optimal policy will be the same.

Let’s look into more detail on the second term in ***M***. After two page of proof in the TRPO paper, we can establish the following lower bound.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*xqv1aAr6DSXnYFSUMWb6bg.jpeg)

The second term in ***M*** is the maximum of KL-divergence underlined in red above. But it is too hard to find and therefore we relax the requirement a little bit by using the mean of the KL-divergence instead. Let’s explain that intuitively.

**Intuition**

*L* approximates the advantage function locally at the current policy. But it gets less accurate as it moves away from the old policy. This inaccuracy has an upper bound. That is the second term in ***M***. After considering the upper bound of this error, we can guarantee that the calculated optimal policy within the trust region is always better than the old policy. If the policy is outside the trust region, even the calculated value may be better but the accuracy can be too off and cannot be trusted. So the bet is off. Let’s summarize the objective below as:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*LH3dbwj7MEYCICAQMzr-zg.jpeg)

Mathematically, both equations above can be resolved to the same optimal policy. However, the theoretical threshold for ***δ*** is very small and is considered to be too conservative. So we relax the condition once more by setting them as tunable hyperparameters.

As mentioned before, ***M*** should be easy to optimize. So we further approximate it to a quadratic equation which is a convex function and heavily study on how to optimize it in high dimensional space.

We use Taylor’s series to expand the terms up to the second-order. But the second-order of 𝓛 is much smaller than the KL-divergence term and will be ignored.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*QKn9GhIMxJmXK_-nAQ8ojA.jpeg)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*A386qkCNen7IXjwfFbH2Og.jpeg)

Therefore, the objective and the constraint can be approximated as:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Sm49fOBxZBvzgSIo8zm-fA.png)

Our objective becomes:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3aO-ncPjkdXwfTEsoik9Hg.png)

We can solve this quadratic equation analytical. The solution is:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*WL22zwP5Sw4v6qmimRY4pA.png)

**What are the challenges?**

## Get Jonathan Hui’s stories in your inbox

Join Medium for free to get updates from this writer.

This solution involves the calculation of the second-order derivative and its inverse, a very expensive operation.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*izpmCUnxEjeIN51cL0RFNQ.png)

So there are two approaches to address this problem:

- Approximate some calculations involving the second order derivative and its inverse to lower the computational complexity, or
- Make the first order derivative solution, like the gradient descent, closer to the second-order derivative solution by adding soft constraints.

TRPO and ACKTR adopt the first approach. PPO is closer to the second one. We can still live with a bad policy decision once a while so we stick with the first-order solution like the stochastic gradient descent. But we are going to add a soft constraint to the objective function so the optimization will have better insurance that we are optimizing within a trust region. So the chance of bad decision is smaller.

## PPO with Adaptive KL Penalty

One way to formulate our objective is changing the constraint to a penalty in the objective function:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bzSagTz4kZxRLuXLviWlxA.jpeg)

*β* controls the weight of the penalty. It penalizes the objective if the new policy is different from the old policy. Borrow a page from the trust region, we can dynamically adjust *β. d* below is the KL-divergence between the old and the new policy. If it is higher than a target value, we shrink *β*. Similarly, if it falls below another target value, we expand the trust region.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*wAuSUKN4z235ogqYpDhQVA.png)

Here is the algorithm:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*NZiUGpkiX5g4g-2yQR6ULg.png)

[Source](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf)

This gets us the performance of TRPO with speed closer to the gradient descent method. But can we do better?

## PPO with Clipped Objective

PPO with clipped objective can even do better. In its implementation, we maintain two policy networks. The first one is the current policy that we want to refine.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*R-yFsRXQFYlZ1qyC-ZmvBQ.png)

The second is the policy that we last used to collect samples.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VVrOkbC4I2qpiZw8hlr5xA.png)

With the idea of importance sampling, we can evaluate a new policy with samples collected from an older policy. This improves sample efficiency.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*OR5Vy1XwhWJQPeZIqE92TA.jpeg)

But as we refine the current policy, the difference between the current and the old policy is getting larger. The variance of the estimation will increase and we will make bad decision because of the inaccuracy. So, say for every 4 iterations, we synchronize the second network with the refined policy again.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vzQrUazL63lk-MGgdqveRw.png)

With clipped objective, we compute a ratio between the new policy and the old policy:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*lEVj3SmB3ja7F8Km7kPB4Q.png)

This ratio measures how difference between two policies. We construct a new objective function to clip the estimated advantage function if the new policy is far away from the old policy. Our new objective function becomes:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7gu7oT4N0HOm_jA0335lVA.png)

If the probability ratio between the new policy and the old policy falls outside the range (1 — ε) and (1 + ε), the advantage function will be clipped. ε is set to 0.2 for the experiments in the PPO paper.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*MpPiARNoNGCxJE2a8m9itA.png)

[Source](https://arxiv.org/pdf/1707.06347.pdf)

Effectively, this discourages large policy change if it is outside our comfortable zone.

Here is the algorithm:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*S8e9USP6x28PLXlu-FA_UQ.png)

[Source](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf)

This new method is simple and can use Gradient Descent like Adam to optimize it. In fact, it is anticlimax for taking so detail analysis on the issue but come up with such a simple solution.

## Thoughts

A quote from the PPO paper:

> Q-learning (with function approximation) fails on many simple problems and is poorly understood, vanilla policy gradient methods have poor data efficiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).

PPO adds a soft constraint that can be optimized by a first-order optimizer. We may make some bad decisions once a while but it strikes a good balance on the speed of the optimization. Experimental results prove that this kind of balance achieves the best performance with the most simplicity.

> Simplicity rules in deep learning.

Or at least until we invent a super fast GPU.

## Credit and references

[PPO paper](https://arxiv.org/pdf/1707.06347.pdf)

[UC Berkeley RL course](http://rail.eecs.berkeley.edu/deeprlcourse/)

[UC Berkeley RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)