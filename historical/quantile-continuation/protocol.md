# Quantile-cycle continuation protocol v1

Frozen before continuation execution. The MDP is exactly the previously certified world 40. No new environment search, reward tuning or initial-state search is allowed. Prior artifacts remain unchanged.

## Questions

1. Does the witnessed exact-backup instability survive larger quantile counts or damping?
2. Does an analogous material control failure occur with actual sampled tabular quantile updates, rather than exact projection?
3. Can the exact witness be strengthened to an open neighborhood of MDP parameters?

The third question is independent mathematical work. Primary-literature priority is checked separately. A positive numerical result does not establish novelty.

## Exact operator experiment

Quantile counts K = 3, 8, 16, 32, 64. Damping alpha = 1, 0.5, 0.1. Initialize all atoms at zero; run 4,000 synchronous updates of z <- (1-alpha)z + alpha*Pi_quantile T_greedy z. Retain the last 512 steps. Identify smallest period 1..128 with sup residual below 1e-8. A strict policy cycle additionally needs distinct greedy policies and minimum action mean gap above 1e-4. Report stable suboptimal fixed policies separately from cycles. Damping the exact projection is not quantile SGD.

## Actual sampled-learning experiment

Tabular quantile Q-learning under a generative simulator: each update independently samples B=16 one-step outcomes for every state-action pair, for all 16 seeds 7100..7115. Each outcome additionally samples one next-critic atom uniformly, giving an unbiased Monte Carlo estimate of the quantile loss over the entire target mixture. Sampled transitions, not exact dynamics, supply training targets. Current mean-greedy action selects the continuation atom. No replay, target network, deep function approximation or oracle target is used.

K = 3, 16, 32. Two losses: exact pinball and quantile Huber with kappa=1. Two per-atom step schedules: constant 0.05; decaying 0.2/(1+t/1000)^0.6. All atoms start at zero. Run 20,000 synchronous sampled updates, checkpoint every 100. Per-atom gradient averages over B target samples, not across output atoms; the effective atom step size is therefore independent of K. Pinball increment is alpha*mean(tau-I[target-z<0]); Huber increment is alpha*mean(abs(tau-I[target-z<0])*clip(target-z,-1,1)). No sorting/reassignment of learned atoms is applied.

Comparator: ordinary scalar Q-learning with the same transition draws, coverage, step schedule and number of updates. It has a separate greedy policy and its own bootstrap values; no access to true transition probabilities. Independent random streams for transition outcomes and target-atom indices preserve outcome pairing across K/loss settings. Both learners are synchronous: all targets use pre-update estimates.

This is actual sampled tabular learning, not an autonomous data-collection experiment and not a QR-DQN reproduction.

## Measurement and decision

Precompute exact values of all eight deterministic stationary policies for evaluation only. Primary sampled endpoint is each seed's mean state-0 stationary-policy regret over checkpoints in the final 20% of updates. Secondary endpoints: final policy, tail fraction of suboptimal policies and checkpoint policy switches. Compare paired per-seed regrets with scalar Q, report mean difference and paired bootstrap 95% interval (10,000 resamples; seed 912). Do not call noisy switching a periodic orbit.

Practical-extension gate: for at least one K>=16, quantile learning must have excess tail regret >=0.25 with positive bootstrap lower bound under BOTH step schedules for the same loss. Otherwise reject the practical extension on this witness/protocol, retaining the exact counterexample. This is a local feasibility gate, not a multiple-comparison-adjusted discovery claim. No rerun with different parameters after failure.

Exact cycles at K>=16 would strengthen the representation-scale extension independently of the sampled gate. Failure at larger K cannot disprove existence of different MDP counterexamples. No prevalence or universal convergence claim is permitted.

## Integrity

Compare empirical generator frequency to its specified law only as implementation sanity, not a research endpoint. Verify gradient signs against direct finite differences away from pinball kinks. Preserve source/protocol/input SHA256, runtime versions, numerical traces, and aggregate results. One exact reproduction is allowed for integrity, not counted as an independent experiment. Scripts must refuse to overwrite completed output directories.
