# Damping and sampled learning on the three-state example

The original K=3 example keeps an exact cycle when the backup is damped by 1/2, and the six-cycle survives small reward and probability perturbations. Sampled learning gave a different result: none of the K≥16 settings reached the tested excess-regret threshold.

## 1. Exact-backup results

The exact-backup comparison uses the same MDP, zero initialization and 4,000 synchronous updates. Damping applies z <- (1-alpha)z + alpha*Pi_quantile T_greedy z. The objective is expected discounted return.

| Quantiles | alpha=1 | alpha=0.5 | alpha=0.1 |
|---|---|---|---|
| 3 | exact-certified 6-cycle | exact-certified 26-cycle | numerical period 127; below the strict-gap threshold |
| 8 | suboptimal fixed point | suboptimal fixed point | suboptimal fixed point |
| 16 | suboptimal fixed point | suboptimal fixed point | suboptimal fixed point |
| 32 | optimal fixed point | optimal fixed point | optimal fixed point |
| 64 | optimal fixed point | optimal fixed point | optimal fixed point |

The K=8 and K=16 fixed policy is (0,0,0), with state-0 stationary-policy regret 0.2736. K=32 and K=64 recover the optimal (0,0,1) policy in these finite runs.

For K=3, alpha=0.5, the rational certifier composes 26 affine damped-backup maps, solves their exact fixed-point equations and checks every quantile backup and greedy choice. It verifies:
- 26 distinct phases and exact closure;
- multiple greedy policies, including the original bad policy;
- minimum greedy gap approximately 0.000138954635677556;
- local one-step contraction factor 19/20;
- minimum sup-norm phase separation approximately 0.01484086197, exceeding twice gap/4.

Reducing the exact-backup step from 1 to 1/2 therefore preserves an attracting policy cycle. At alpha=0.1, the numerical gap is about 0.00000670848, below the 0.0001 strict-cycle reporting threshold; that numerical orbit has no exact certificate.

## 2. Sampled tabular learning

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

The paired scalar baseline has mean final-window regret 0.074385 with the constant schedule and 0.0636975 with the decaying schedule. Negative excess means the quantile learner has lower regret than scalar control.

The criterion was excess regret at least 0.25 with a positive lower interval endpoint under both schedules for the same loss at K≥16. No setting passed. The K=3 pinball effect survives sampled learning, but Huber smoothing sharply reduces it.

These intervals describe seed variability on this selected MDP. The generative sampler tests tabular learning with full state-action coverage; it does not test exploration or replay.

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

In this box, orbit displacement is bounded by 1/1000 and the perturbed true optimal-action gap has conservative lower bound 3/125. The [rational checker](certify_robustness.py) verifies these margins and inequalities. The Lipschitz argument above supplies local attraction within this fixed transition topology and discount.

## 4. Relation to the general construction

This fixed MDP gives a locally attracting finite-quantile cycle with strict greedy choices, unique quantiles and a suboptimal phase. The damped certificate and open reward/probability neighborhood strengthen that result. The later [arbitrary-K construction](../../THEOREM.md) uses a different MDP for each K; it does not change this capacity experiment's outcome. [Related work](../../PRIOR_ART.md) compares finite-quantile suboptimality and distributional-cycle precedents.

## 5. Reproduction

The files alongside this report include:

- `protocol.md`: continuation settings and stopping criterion.
- `run.py`: exact-backup grid and sampled learners.
- `results-v1/`: policy traces, final critics, results and runtime metadata.
- `reproduction-v1/`: a second run with the same seeds.
- `certify_damped.py` and `damped-certificate.json`: exact period-26 cycle.
- `certify_robustness.py` and `robustness-certificate.json`: open-neighborhood bounds.
- `verify.py` and `verification.json`: input/output hashes and comparison of the two runs.

The original runtime was Python 3.14.7 and NumPy 2.3.5. Each full run took approximately 26 seconds on CPU. The saved verification checked 32 output-file hashes, 41 array members and 3 JSON comparisons. Values matched across runs, excluding wall time and archive metadata.

The verifier also solves all eight policy Bellman systems with rational arithmetic and recomputes all 12 sampled endpoints from the policy traces.

Each run generated 30,720,000 one-step transition outcomes, reused across methods. Quantile learners consumed 368,640,000 transition samples and the two scalar baselines consumed 61,440,000. The second run repeats the same random outcomes.

Reproduce with a NumPy-enabled Python interpreter:

    python run.py --output a-new-output-directory
    python certify_damped.py
    python certify_robustness.py
    python verify.py

The certifiers and verifier use the stored relative input paths. `verify.py` compares `results-v1` with `reproduction-v1`; `run.py` refuses to overwrite an existing output directory.

## Interpretation

- Exact cycles persist at K=3 for alpha=1 and alpha=1/2, with local attraction.
- Small reward and probability perturbations preserve the original six-cycle and its suboptimal phase.
- Sampled pinball has substantial excess regret at K=3; no K≥16 setting meets the 0.25 excess-regret criterion.
- Finite sampled switching does not establish an asymptotic cycle, and these runs contain no deep function approximation.
