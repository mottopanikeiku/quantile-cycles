# A strict six-cycle in a three-state MDP

Exact finite-quantile Bellman control on this MDP has a locally attracting six-cycle with strict greedy choices. One phase selects a suboptimal stationary policy. The [continuation study](quantile-continuation/REPORT.md) adds a damped 26-cycle and open-parameter robustness, then tests sampled learning at larger capacities.

## Exact problem and operator

There are 3 observed states, 2 actions, 2 stochastic outcomes per state-action pair, and discount gamma=9/10. Each outcome has probability 2/5 or 3/5 respectively. Rewards and next states are:

| State | Action | Outcome with probability 2/5: (reward,next) | Outcome with probability 3/5: (reward,next) |
|---|---|---|---|
| 0 | 0 | (0,2) | (1,1) |
| 0 | 1 | (0,0) | (-2,1) |
| 1 | 0 | (3,2) | (-2,1) |
| 1 | 1 | (1,2) | (-3,0) |
| 2 | 0 | (-1,2) | (0,0) |
| 2 | 1 | (-3,1) | (2,1) |

Represent each state-action return law by three equally weighted locations z(s,a,i). At each synchronous iteration:

1. Select pi(s)=argmax_a mean_i z(s,a,i).
2. Form the exact six-atom law of R+gamma*z(S',pi(S'),i).
3. Replace its distribution with generalized-inverse quantiles at 1/6, 1/2 and 5/6.

The objective is expected discounted return. The backup uses the exact target law, with no sampled gradient, replay or function approximation.

## Search and exact certificate

The search used seed 418, 512 random MDPs, 600 iterations and zero initialization. The first qualifying witness was world index 40. Candidate periods were 2 through 8, and the last 64 iterates had to satisfy strict action-gap and periodicity checks.

The standard-library rational certifier infers local affine routing from the numerical orbit, solves the period-composed linear equations exactly and reapplies all six quantile backups using Fractions. It checks each inferred route, exact closure and six distinct phases.

Results:
- Exact period: **6**.
- Phase policies: (0,0,1), (0,1,1), (0,0,1), (0,0,1), (0,0,1), (0,0,1).
- Minimum greedy action gap across all states and phases: **52729/5000000 = 0.0105458**.
- Unique quantiles: every target mass is a multiple of 1/15, while requested fractions are odd multiples of 1/6. No requested quantile lies on a CDF plateau boundary.
- Optimal stationary policy: (0,0,1), with values **(3/5,0,0)**.
- Bad phase's stationary policy: (0,1,1), with values **(-5025/1309,-6725/1309,-12105/2618)**.
- State-0 stationary-policy loss in the bad phase: **29052/6545 = 4.43880825057**.

The policy loss evaluates a frozen phase policy for the infinite-horizon discounted task, rather than a controller that changes its policy at every environment step.

## Local attraction argument

Let g be the exact minimum action gap above. In a sup-norm ball of radius g/4 around any phase, each action mean changes by at most g/4, so its greedy action remains unchanged. With this fixed policy, each target atom changes by at most gamma times the input perturbation. Quantiles of the coupled discrete laws therefore change by at most the same amount. The backup sends the phase ball into the next phase ball with contraction factor gamma.

The six-step composition maps the first phase ball into itself and contracts by gamma^6=531441/1000000. Together with exact closure, this establishes a locally attracting periodic orbit with radius 52729/20000000. Zero initialization reaches the orbit numerically; the attraction proof is local.

## Scalar-control baseline

The comparator maintains six ordinary scalar Q entries for Bellman optimality and uses their greedy policy for the same quantile backups. On the witness, after 600 iterations both scalar and distributional residuals were numerically zero, all final 64 policies were optimal, and there were no policy switches.

This separates scalar risk-neutral control from distributional evaluation. A mean-preserving categorical projection is another relevant baseline when its support contains the targets.

## Related work

- [Bellemare, Dabney and Rowland, Distributional Reinforcement Learning, Chapter 7](https://www.distributional-rl.org/contents/chapter7) describes distributional control nonconvergence and scalar control followed by distributional evaluation.
- [Rowland et al., Statistics and Samples in Distributional Reinforcement Learning](https://arxiv.org/abs/1902.08102) gives finite-quantile suboptimality examples.
- [Kuang et al., Variance Control for Distributional Reinforcement Learning](https://proceedings.mlr.press/v202/kuang23a.html) introduces Quantiled Expansion Mean to improve scalar Q estimates from quantile critics.
- [Q-learning for Quantile MDPs](https://arxiv.org/abs/2410.24128) studies quantile-objective control. This example instead selects actions by the mean of the represented distribution.

The distinguishing feature here is a locally attracting cycle with strict greedy choices, unique quantiles and a suboptimal phase. [PRIOR_ART.md](../PRIOR_ART.md) compares these properties in more detail.

## Files and commands

The files are in this directory:

- `quantile-cycle-protocol.md`: search settings and the choice of outcome probabilities.
- `quantile_cycle.py` and `quantile-cycle-result.json`: NumPy search and witness.
- `certify_quantile_cycle.py` and `quantile-cycle-certificate.json`: exact orbit, greedy gaps and contraction information.
- `quantile_control_baseline.py` and `quantile-control-baseline-result.json`: scalar-control comparator.

The search and comparator require NumPy. From this directory, `python3 certify_quantile_cycle.py` checks the exact witness with the standard library. See the [historical README](README.md) for the combined checks.

The subsequent [continuation report](quantile-continuation/REPORT.md) covers larger critics, damping, sampled updates and parameter perturbations. It found no material sampled excess regret at K≥16 on this fixed MDP.
