# Quantile-control continuation: theorem strengthened, practical extension rejected

## Decision

Retain the result as a narrow theoretical/mechanistic candidate. **Reject the practical-instability extension on this witness under continuation protocol v1.** Do not launch a new controller project from these results, claim QR-DQN nonconvergence, or broaden the tested claim after its failed gate.

This continuation adds actual sampled tabular learning, an exact damped-cycle certificate, and an open-neighborhood theorem. It does not establish a new algorithm, empirical superiority, or theoretical priority.

## 1. Exact-backup results

Same MDP, zero initialization, 4,000 synchronous updates, no environment search. The objective remains expected discounted return. Damping means z <- (1-alpha)z + alpha*Pi_quantile T_greedy z; it is not quantile-loss SGD.

| Quantiles | alpha=1 | alpha=0.5 | alpha=0.1 |
|---|---|---|---|
| 3 | exact-certified 6-cycle | exact-certified 26-cycle | numerical period 127; misses frozen strict-gap threshold |
| 8 | suboptimal fixed point | suboptimal fixed point | suboptimal fixed point |
| 16 | suboptimal fixed point | suboptimal fixed point | suboptimal fixed point |
| 32 | optimal fixed point | optimal fixed point | optimal fixed point |
| 64 | optimal fixed point | optimal fixed point | optimal fixed point |

The K=8 and K=16 fixed policy is (0,0,0), with state-0 stationary-policy regret 0.2736. K=32 and K=64 recover the optimal (0,0,1) policy. These are numerical convergence observations under the specified finite run, not a general convergence theorem.

For K=3, alpha=0.5, a new rational certifier composes 26 affine damped-backup maps, solves their exact fixed-point equations, and independently checks each exact quantile backup and greedy choice. It verifies:
- 26 distinct phases and exact closure;
- multiple greedy policies, including the original bad policy;
- minimum greedy gap approximately 0.000138954635677556;
- local one-step contraction factor 19/20;
- minimum sup-norm phase separation approximately 0.01484086197, exceeding twice gap/4.

Therefore reducing the exact-backup step from 1 to 1/2 does not eliminate this example's attracting policy cycle. At alpha=0.1, the numerical gap is about 0.00000670848, below the predeclared 0.0001 strict-cycle reporting gate; no exact certificate was attempted for that numerical orbit.

## 2. Actual sampled tabular-learning results

Training used sampled one-step transitions from a generative simulator. Every update sampled 16 targets per state-action pair. Continuation target atoms were uniformly sampled from each learner's own current critic. There were 16 independent seed labels, 20,000 updates per setting, and paired transition draws across quantile/scalar methods. Both methods used only sampled training targets; exact policy values were used solely for evaluation.

The loss is either pinball or quantile Huber (kappa=1). Per-atom step sizes are 0.05 or 0.2/(1+t/1000)^0.6. The primary endpoint averages each seed's state-0 stationary-policy regret over checkpoints in the last 20% of training. Positive excess regret means quantile learning was worse than scalar Q-learning.

| K | Loss | Constant-step excess regret [95% paired bootstrap interval] | Decaying-step excess regret [95% interval] |
|---|---|---|---|
| 3 | Pinball | 0.66224 [0.59219, 0.73621] | 0.64660 [0.58083, 0.72197] |
| 3 | Huber | 0.04104 [0.03035, 0.05173] | 0.05301 [0.03762, 0.06926] |
| 16 | Pinball | -0.02779 [-0.03976, -0.01538] | 0.02394 [0.00855, 0.03890] |
| 16 | Huber | -0.04403 [-0.05600, -0.03249] | -0.04061 [-0.05429, -0.02779] |
| 32 | Pinball | -0.04532 [-0.05942, -0.03164] | -0.04703 [-0.06071, -0.03377] |
| 32 | Huber | -0.04275 [-0.05600, -0.02993] | -0.04190 [-0.05558, -0.02822] |

The paired scalar baseline itself has mean tail regret 0.074385 with the constant schedule and 0.0636975 with the decaying schedule. A negative difference is therefore possible and does not represent negative absolute regret. In particular, larger quantile critics usually did slightly better here.

The frozen practical gate required excess regret at least 0.25 with positive lower interval endpoint under both schedules for the same loss at K>=16. **No cell passed.** The K=3 pinball effect survives sampled learning, but standard Huber smoothing sharply reduces it, and it does not support the predeclared larger-critic extension.

These intervals describe seed variability on one selected MDP. They do not account for environment selection or justify an across-task claim. Checkpoint switching under sampled updates is not evidence of a periodic orbit. This is tabular generative-model learning, not autonomous exploration, replay-based deep QR-DQN, or an asymptotic stochastic convergence result.

## 3. Open-neighborhood theorem

The original exact six-cycle is not restricted to the exact reward/probability numbers in the witness. Keep transition destinations, discount 9/10, K=3, midpoint quantile levels and uniform continuation-atom weights fixed.

Let delta bound the absolute perturbation of each of the 12 outcome rewards, and eta bound each of the six first-outcome probabilities away from 2/5 (the second probability remains complementary).

### Probability invariance

Every target subset CDF mass is (2j+3l)/15 for j,l in {0,1,2,3}. Each requested quantile lies at an odd multiple of 1/6, so its distance to any such mass is at least 1/30. Under a probability perturbation dp, the subset mass changes by (j-l)dp/3, whose magnitude is at most |dp|.

Thus for **eta<1/30**, every quantile-boundary sign is unchanged, for every atom vector and ordering. With unchanged rewards, the projected operator is exactly identical globally, including at coincident atom values. This statement concerns the projected operator, not the true MDP's optimal values.

### Reward robustness and local attraction

The certified original minimum greedy gap is g=52729/5000000. Set rho=g/4. The six phase balls of radius rho are disjoint: their minimum center separation is 1635243957/2466100000, much larger than 2rho.

A fixed-policy projected quantile backup is gamma-Lipschitz in atom sup norm. Reward perturbations change its output by at most delta. Consequently **delta<(1-gamma)rho=52729/200000000**, together with the probability condition above, gives cyclic phase-ball invariance. The six-step map contracts by gamma^6 and has a unique attracting six-cycle in those balls. Its phase displacement is at most delta/(1-gamma); the greedy policy sequence is unchanged.

### Preserving genuine suboptimality

At the baseline optimal value V*=(3/5,0,0), the largest two-outcome return span is 5. The baseline minimum true optimal-action gap is h=19/250. The additional sufficient condition

    delta + 5*eta < h*(1-gamma)/2 = 19/5000

preserves the same unique optimal policy. It therefore preserves the bad phase's genuine suboptimality as well.

One explicit open box satisfying all conditions is:
- every outcome reward changes by less than **1/10000**;
- every first-outcome probability changes by less than **1/2000**.

In this box, orbit displacement is bounded by 1/1000 and the perturbed true optimal-action gap has conservative lower bound 3/125. Exact rational arithmetic verifies the margins and inequalities; an independent mathematical review found no correctness issue. The arithmetic checks support the proof but do not substitute for its Lipschitz/invariance argument.

This is a local theorem within a fixed transition topology and discount. It makes no global-attraction or practical deep-learning claim.

## 4. Closest prior results and novelty status

The reviewed primary literature brackets this result closely:

- Rowland et al., *Statistics and Samples in Distributional Reinforcement Learning*, Appendix B.4: https://arxiv.org/abs/1902.08102 . For every finite K, a one-step example gives a strict suboptimal QDRL choice with unique midpoint quantiles: a rare Bernoulli reward has true mean 1/(4K) but projected mean zero, while a deterministic action pays 1/(8K). This already defeats any claim that strict finite-quantile suboptimality itself is new. It is not a nontrivial attracting cycle.
- Bellemare, Dabney and Rowland, *Distributional Reinforcement Learning*, Chapter 7, Example 7.11: https://www.distributional-rl.org/contents/chapter7 . Distributional optimality operators can cycle through distribution-dependent tie-breaking among equal-mean optimal actions. General distributional nonconvergence is not new.
- The same chapter explains mean-preserving projections and scalar-control/distributional-evaluation separation. These remain existing remedies, not proposed algorithms.
- Kuang et al., *Variance Control for Distributional Reinforcement Learning*: https://proceedings.mlr.press/v202/kuang23a.html . Quantiled Expansion Mean is a close practical alternative for estimating scalar control values from a quantile critic.
- Rowland et al., *An Analysis of Quantile Temporal-Difference Learning* (JMLR 2024): https://www.jmlr.org/papers/v25/23-0154.html . Its stochastic convergence analysis is for policy evaluation, not the moving mean-greedy control map here. In particular, the present full-backup cycles do not contradict that evaluation result.

The narrower uncollided statement in the reviewed sources is a **locally attracting finite-quantile cycle with strict greedy choices, unique quantiles and a strictly suboptimal phase**, now also exact-certified at damping 1/2 and robust on an open MDP-parameter neighborhood. This is not proof of priority. The generalization to arbitrary quantile counts is not established by this work.

## 5. Reproduction and integrity

All continuation files are alongside this report. Preserve both runs:

- protocol.md — frozen continuation design and gate.
- run.py — exact grid and actual sampled learners.
- results-v1/ — original results, full tail exact arrays, sampled policy traces, final critics, source/input/output hashes and runtime metadata.
- reproduction-v1/ — one exact integrity reproduction, not an independent experiment.
- certify_damped.py and damped-certificate.json — exact period-26 proof.
- certify_robustness.py and robustness-certificate.json — exact open-neighborhood bounds.
- verify.py and verification.json — source/output bindings and semantic reproduction check.

The launch runtime was Python 3.14.7 and NumPy 2.3.5. Each full run took approximately 26 seconds on CPU. Verification checked 32 output-file hashes, 41 array members and 3 JSON comparisons across the two runs; semantic outputs matched exactly apart from deliberately excluded wall time and container-file hash metadata.

An independent exact-rational audit also solved all eight policy Bellman systems and recomputed all 12 sampled endpoints from the retained policy traces. Those checks matched the recorded numerical scores.

Each run generated 30,720,000 one-step transition outcomes, reused across methods; quantile learners consumed 368,640,000 transition samples and the two scalar baselines consumed 61,440,000. These are generative-model samples, not long online episodes. Reproduction repeats the same random outcomes and supplies no extra statistical evidence.

Reproduce with a NumPy-enabled Python interpreter:

    python run.py --output a-new-output-directory
    python certify_damped.py
    python certify_robustness.py
    python verify.py

The certifiers and verifier use the preserved relative input paths; verify.py compares results-v1 with reproduction-v1. run.py refuses to overwrite an existing output directory. No existing project was changed and no public successor repository was created.

## Final scope

**Supported:** exact strict cycles; persistence under one nontrivial damping setting; local attraction; open reward/probability robustness; a substantial sampled pinball effect at K=3.

**Rejected under this protocol:** material sampled instability at K>=16 on this witness; exact-cycle persistence at the tested larger K values.

**Unestablished:** novelty priority, arbitrary-K cycle constructions, deep-RL relevance, prevalence, a new remedy, or a broad practical research thesis. The deliverable is now a stronger bounded mathematical result with preserved negative practical evidence—not a successful new RL algorithm project.
