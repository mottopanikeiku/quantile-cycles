# Quantile-loss control on the analytic family

I fixed this protocol and `sampled-config.json` before running the study. I will not change the MDP, seeds, schedules or endpoints in response to the results. This is a finite synthetic learning experiment, separate from the exact hard-projection theorem.

## Stages and stopping rule

1. I first enumerate the finite target distribution and run hard projection, damped hard projection, expected pinball gradients, expected quantile-Huber gradients with κ=1, and expected scalar Q-learning. I use zero initialization and 100,000 synchronous updates for every method. Hard projection has no step-size schedule; damped projection and gradient methods use both schedules below.
2. I then run the full tabular sampled comparison: 32 seeds, 100,000 updates per seed and 32 independently sampled reward/continuation pairs per action per update. I run both pinball and Huber and a scalar baseline regardless of whether population Huber keeps the hard cycle. This distinguishes smoothing from finite-sample noise without picking favorable cells.
3. I run the replay/network stage only if **all four** family Huber cells (K=8 and K=32, both schedules) have population normalized excess regret at least 0.25 **and** sampled normalized excess regret at least 0.25 with a strictly positive paired 95% bootstrap lower endpoint. If either stage fails this rule, I stop before replay/network training and report the failure, not a deep-RL conclusion.

The JSON configuration is the complete runnable specification. The only allowed run splitting is by case, method or seed batch. A short pilot uses the same implementation with fewer updates and is not part of the final endpoint. I stop starting experiments at 08:00 PDT on 7 October 2026. A compute failure is reported as missing data; it is not a reason to drop cells or replace their endpoint.

## Environments and updates

The family uses K=2, 8 and 32 with discount β=1/(2K), ε=1/(4K²), B rewards i/K and weights `(1/(2K)-ε, 1/K, …, 1/K, 1/(2K)+ε)`. A pays `c=(K+1)/(2K)-β(1+β)(K-1)/(4K)`. These are three different MDPs. The separately labeled fixed-MDP comparison uses the K=2 rewards and β=1/4 at critic capacities 2, 8 and 32. Family-2 and fixed-2 are the same numerical problem; their labels are retained, not treated as independent replications.

I select the continuation action by the mean of the pre-update critic; A wins exact ties. Targets are `reward + β * uniformly selected continuation atom`. I update both actions synchronously. The per-atom step schedules, with update index t starting at zero, are `0.05` and `0.2/(1+t/1000)^0.6`. A gradient averages over the target law or the 32 samples, not over the number of output atoms. Pinball uses subgradient zero at equality. Huber uses `|τ-I(target<atom)| * clip(atom-target, -κ, κ)/κ`; targets are detached from the gradient.

Each seed uses the same reward and continuation uniforms across quantile losses and schedules. Scalar Q-learning uses the paired reward draws and its own scalar maximum continuation; its update uses the batch-mean Bellman target. Counter-based randomness avoids a loss-dependent number of draws. Pairing is within a seed, not a claim that critic trajectories are identical. No exploration or replay is used in the tabular stage.

## Endpoints and uncertainty

The primary endpoint uses the post-update greedy actions at t=80,001,…,100,000. A's stationary policy has regret zero and always B loses `(c-(1/2+ε))/(1-β)`. Therefore mean stationary-policy regret divided by this known loss equals the fraction of these 20,000 greedy actions that are B. This evaluates the policy frozen at each checkpoint/update; it is not the value of the changing training-time policy.

I report the normalized endpoint, absolute endpoint, and paired excess over scalar control for every cell and all seeds. I use 10,000 percentile-bootstrap resamples of the 32 seed pairs with RNG seed 20261007 for 95% intervals. Seed pairs, not individual updates, are the resampling unit. The population stage is deterministic and has no seed interval. I separately report final-window switching fraction; a noisy switching policy is not proof of a periodic orbit.

At the fixed checkpoints in the configuration I save both critic means, greedy action, exact hard projected Bellman residual, and expected pinball/Huber losses. The hard-quantile CDF margin ε shrinks as K grows. I report its size against the binomial CDF standard error for batch 32 and the independent-sample count needed for a 1.96-standard-error margin at the central quantile. Repeated updates with changing targets do not count as an equivalent fixed-target sample batch.

## Conditional network stage

If the gate passes, I use the two gate environments, the same 32 seeds and 100,000 environment steps. A one-hot one-state input feeds a 16-unit ReLU hidden layer and 2K quantile outputs; scalar DQN uses the same hidden layer and two outputs. I use uniform replay capacity 10,000, 1,000 warm-up steps, minibatch 32, Adam learning rate 0.001, epsilon decreasing from 1 to 0.05 over 10,000 steps, and target refresh 100. Refreshes 1 and 1,000 are also fixed mechanism comparisons, never replacements chosen after seeing the result. Exploration/reward uniforms are paired with scalar DQN. I use the same primary final-20% policy endpoint and seed-bootstrap comparison. The network is a synthetic optimization/replay check, not a generalization test.

Written with AI coding assistance.
