# Theory
## PPO vs TRPO
In this project we use the Reinforcement Learning algorithm called PPO (Proxpimal Policy Optimization), algorithm made by OpenAI in 2017 and it is now a standard in reinforcement learning. This algorithm is a derivation from TRPO (Trust Region Policy Optimization), with PPO being a faster approximation of TRPO.

I struggled to find a simple/TLDR explanation about what do these algorithms consist of, so I will try to do it now.

## TRPO
Before learning about PPO, it makes more sense to learn about TRPO.

### Problem
TRPO and PPO are made to solve reinforcement learning problems, by optimizing a policy

Given a state $S$, a policy $\pi$ select an action $a$ with some probability, so $P(a|S)$.<br>
In our case, a policy will select a Pokémon move or a switch given a battle state.

A Policy is often decided by a model (here a Neural Network) with a set of parameters `θ`
So our relation become $\pi_{\theta}(S) = a$.

Each of our action affect the state, and at the end, we attribute to the final state a value, the Reward `R`
So for multiple choices:
* Policy selects action 0 on the state 0 : $\pi_{\theta}(S_0) = a_0$
* The state 0 becomes state 1 thanks to the action : $S_0(a_0) = S_1$
* Policy selects a new action: $\pi_{\theta}{S_1} = a_1$
* States updates: $S_1(a_1) = S_2$
* ...
* At the end, we attribute the final state a reward value: $reward(S_n) = R$

**Our goal is to find which policy $\theta$ yields the best reward R, on average**

## Concept of TRPO
TRPO updates its policy using the KL-Divergence to make sure the update are constrained and not too much.
In Reinforcement Learning, it was very common that model learned too much, because it's hard
to gauge how big an impact a policy change can have. Previous algorithms pushed those changes too much
in the direction that maximize the expected reward. 

The KL-Divergence (Kullback–Leibler divergence) is a way to measure the distance[^1] between two policies.

**TRPO says, when we update a policy, we do not update it too much.**

## Algorithm
* We run a policy $pi_{\theta}$ over a batch of runs, collecting for each the reward $R$
* Now we ask, how can I change $\theta$ to maximize $R$ ?
  * BUT we need to keep the KL-Divergence of $P_{\theta}$ and $P_{\theta'}$ (our new policy) small (often < 0.02)
* Once we have found our best $\theta'$, we set it as our baseline policy and we loop back on the first step.

This restriction on "how far can the new policy be from our baseline" is the source of the name "Trust Region", the trust region is any policy for which the divergence with our baseline is small enough.
## The problem
While KL-Divergence is easy to compute, finding what $\theta'$ satisfy the condition is computationnaly hard.

## PPO
The goal of PPO is to find boundaries for the policy changes that are easier to compute than TRPO but effectively does the same thing. 

## Ratio
Instead of saying "KL-Divergence must stay low", PPO says "Ratio between policies must stay close to 1"
For each step $t$, the ratio is $r_t(\theta,\theta') = \frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)}$.
Then for each step we also estimate the advantage $A_t$ i.e. how much better or worse taking action $a_t$ was compared to what we expected from state $s_t$.

## Clipping
For each probability, in the objective function, we clip the ratio betwen $1-\epsilon$ and $1+\epsilon$ so $\pi_{\theta'}$ doesn't have incentives to go too far from $\pi_{\theta}$.<br>
In other word, if the ratio is greater than $1+\epsilon$, it is as if the ratio is $1+\epsilon$ and the model doesn't gain any benefit of pushing the ratio further.

In practice, for it to work with negative advantage, our objective function must look like $min(r_t * A_t, clip(r_t, 1-\epsilon, 1+\epsilon) * A_t)$
This is the function we want to optimize, i.e. find the max of.

[^1] Literally not a distance, big woop.
