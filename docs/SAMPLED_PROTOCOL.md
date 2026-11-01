# Quantile-loss control on the analytic family

The study compares exact projection, population gradients and sampled learning on the MDP family in [THEOREM.md](../THEOREM.md). [sampled-config.json](../sampled-config.json) contains the settings; [SAMPLED_RESULTS.md](SAMPLED_RESULTS.md) reports the outcome.

## Methods and stopping rule

1. Population updates enumerate the finite target distribution. Methods are hard projection, damped hard projection, expected pinball gradients, expected quantile-Huber gradients with κ=1 and expected scalar Q-learning. All start at zero and run for 100,000 synchronous updates. Hard projection has no step-size schedule; the other methods use both schedules below.
2. Sampled updates use 32 seeds, 100,000 updates per seed and 32 independently sampled reward/continuation pairs per action per update. Pinball, Huber and scalar control run under both schedules, regardless of the population result.
3. The replay/network experiment requires all four family Huber comparisons (K=8 and K=32, both schedules) to have population normalized excess regret at least 0.25 and sampled normalized excess regret at least 0.25 with a strictly positive paired 95% bootstrap lower endpoint. None met this condition, so that experiment was not run.

Runs may be split by case, method or seed batch. Pilots use the same implementation with fewer updates and are excluded from the final endpoint.

## Environments and updates

The family uses K=2, 8 and 32 with discount β=1/(2K), ε=1/(4K²), B rewards i/K and weights `(1/(2K)-ε, 1/K, …, 1/K, 1/(2K)+ε)`. A pays `c=(K+1)/(2K)-β(1+β)(K-1)/(4K)`. These are three different MDPs. The fixed-MDP comparison uses the K=2 rewards and β=1/4 at critic capacities 2, 8 and 32. Family-2 and fixed-2 are the same numerical problem under two labels.

The continuation action maximizes the mean of the pre-update critic; A wins exact ties. Targets are `reward + β * uniformly selected continuation atom`. Both actions update synchronously. The per-atom step schedules, with update index t starting at zero, are `0.05` and `0.2/(1+t/1000)^0.6`. A gradient averages over the target law or the 32 samples, without averaging over output atoms. Pinball uses subgradient zero at equality. Huber uses `|τ-I(target<atom)| * clip(atom-target, -κ, κ)/κ`; targets are detached from the gradient.

Each seed uses the same reward and continuation uniforms across quantile losses and schedules. Scalar Q-learning uses the paired reward draws and its own scalar maximum continuation; its update uses the batch-mean Bellman target. Counter-based randomness keeps the draw count independent of the loss. The tabular experiment uses a generative sampler with no exploration or replay.

## Endpoints and uncertainty

The primary endpoint uses post-update greedy actions at t=80,001,…,100,000. A's stationary policy has regret zero and always B loses `(c-(1/2+ε))/(1-β)`. Mean stationary-policy regret divided by this loss equals the fraction of these 20,000 greedy actions that are B. This evaluates each greedy policy as a stationary policy, rather than the return of a policy that changes during training.

For every setting, the summary includes normalized regret, absolute regret and paired excess over scalar control. The 95% intervals use 10,000 percentile-bootstrap resamples of the 32 seed pairs with RNG seed 20261007. The population stage is deterministic. Final-window switching fractions count transitions within that window.

At the configured checkpoints, the results include both critic means, greedy action, exact hard-projected Bellman residual and expected pinball/Huber losses. The hard-quantile CDF margin ε shrinks with K. Its sampling diagnostic compares it with the binomial CDF standard error for batch 32 and computes the independent fixed-target sample count needed for a 1.96-standard-error margin at the central quantile. Repeated updates with changing targets do not constitute an equivalent fixed-target sample batch.

## Unrun network experiment

The conditional design used the K=8 and K=32 family environments, the same 32 seeds and 100,000 environment steps. A one-hot one-state input would feed a 16-unit ReLU hidden layer and 2K quantile outputs; scalar DQN would use the same hidden layer and two outputs. Settings were uniform replay capacity 10,000, 1,000 warm-up steps, minibatch 32, Adam learning rate 0.001, epsilon decreasing from 1 to 0.05 over 10,000 steps and target refresh 100. Target refreshes 1 and 1,000 were additional comparisons. Exploration/reward uniforms would be paired with scalar DQN, using the same final-20% policy endpoint and seed-bootstrap comparison. This design tests synthetic optimization and replay, with no generalization task.
