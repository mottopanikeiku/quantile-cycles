# Successor candidate: strict policy cycles in quantile control

**Continuation update:** [The completed continuation report](quantile-continuation/REPORT.md) strengthens the exact theorem with a damped 26-cycle and open-parameter robustness, but rejects the larger-critic practical extension under its frozen gate. The text below preserves the initial candidate decision and evidence; use the continuation report for current scope.

## Decision

CONTINUE this narrow theoretical/mechanistic candidate. Do not claim a new controller, benchmark improvement, or established priority. Unlike the preceding screens, this candidate has an executed discriminator and an exact rational witness rather than only a plausible failure story.

**Thesis:** finite-quantile projection can turn risk-neutral distributional Bellman control into a locally attracting strict policy cycle that repeatedly selects a genuinely suboptimal policy, even with exact stationary dynamics, unique projected quantiles, and unique greedy actions. The research target is characterizing when this happens and its relevance to practical learning, not inventing scalar-value control as a remedy.

This is a stronger *supported candidate*, not yet a demonstrated publishable advance. A close prior counterexample or a failure to extend beyond tiny full-backup critics could reduce it to an educational note.

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

This is finite-quantile projected risk-neutral distributional optimality iteration. It is not risk-sensitive quantile-objective control. There is no sampled reward estimate, gradient descent, target network, replay, or function approximation.

## Executed evidence

The frozen search used seed 418, 512 random MDPs, 600 iterations and zero initialization. The first qualifying witness was world index 40. Search periods were fixed to 2 through 8, and the last 64 iterates had to satisfy strict action-gap and periodicity checks. The search was reproduced once to persist the result.

A separate standard-library rational certifier inferred the local affine routing from the numerical orbit, solved its period-composed linear equations exactly, then independently reapplied all six quantile backups using Fractions. Every inferred route was checked against the exact backup; exact closure and six distinct phases were verified. The numerical output is not itself the proof.

Results:
- Exact period: **6**.
- Phase policies: (0,0,1), (0,1,1), (0,0,1), (0,0,1), (0,0,1), (0,0,1).
- Minimum greedy action gap across all states and phases: **52729/5000000 = 0.0105458**.
- Unique quantiles: every target mass is a multiple of 1/15, while requested fractions are odd multiples of 1/6. No requested quantile lies on a CDF plateau boundary.
- Optimal stationary policy: (0,0,1), with values **(3/5,0,0)**.
- Bad phase's stationary policy: (0,1,1), with values **(-5025/1309,-6725/1309,-12105/2618)**.
- State-0 stationary-policy loss in the bad phase: **29052/6545 = 4.43880825057**.

The policy loss evaluates a frozen phase policy for the infinite-horizon discounted task. It is not the return of a controller switching its policy every environment step, and it is not an average training score.

## Local attraction argument

Let g be the exact minimum action gap above. In a sup-norm ball of radius g/4 around any phase, each action mean changes by at most g/4, so its greedy action remains unchanged. With this fixed policy, each target atom changes by at most gamma times the input perturbation. Quantiles of the coupled discrete laws therefore change by at most the same amount. The backup sends the phase ball into the next phase ball with contraction factor gamma.

The six-step composition maps the first phase ball into itself and contracts by gamma^6=531441/1000000. Together with exact closure, this establishes a local attracting periodic orbit, not merely a floating-point oscillation or a greedy tie artifact. The certified radius is **52729/20000000**. Global attraction is not claimed; zero initialization reaching this orbit was observed numerically.

## Closest baseline: it already fixes the control problem

The comparator maintains six ordinary scalar Q entries for Bellman optimality and uses their greedy policy for the same quantile backups. On the witness, after 600 iterations both scalar and distributional residuals were numerically zero, all final 64 policies were optimal, and there were no policy switches.

This is a known separation of risk-neutral control from distributional evaluation. It costs six scalar entries here and immediately defeats any claim that a new sophisticated controller is needed for this family. A mean-preserving categorical projection is another relevant existing baseline when its support contains the targets. Neither remedy is a proposed invention.

## Literature collision and remaining distinction

- Bellemare, Dabney and Rowland, *Distributional Reinforcement Learning*, Chapter 7: https://www.distributional-rl.org/contents/chapter7 . Section 7.3 explains equivalence to scalar value iteration under mean preservation and explicitly describes scalar control followed by distributional evaluation. Remark 7.2 bounds quantile-projection value error. General distributional nonconvergence is already known; it is not the proposed contribution.
- Rowland et al., *Statistics and Samples in Distributional Reinforcement Learning*: https://arxiv.org/abs/1902.08102 . Finite quantiles are not Bellman-closed. Discovering projection bias alone is not new.
- Kuang et al., *Variance Control for Distributional Reinforcement Learning*: https://proceedings.mlr.press/v202/kuang23a.html . Quantiled Expansion Mean directly targets the quality of quantile-derived scalar Q estimates; it is a close algorithmic competitor.
- *Q-learning for Quantile MDPs*: https://arxiv.org/abs/2410.24128 . Quantile-objective control and nonunique set-valued quantile operators are adjacent but different: this witness is mean-greedy and its quantiles and greedy actions are unique.

The narrower candidate is a **locally attracting strict suboptimal-policy cycle caused by finite projection**, rather than oscillation among equally optimal distributions or nonunique quantiles. Targeted searches did not establish whether this exact statement is new. Search absence is not a priority proof.

## Other lanes screened in this continuation

Recurrent replay: finite burn-in cannot erase error in deliberately retained memory directions, but that observation is established. Recent burn-in analysis (https://arxiv.org/abs/2602.10911) and recurrent sensitivity correction (https://arxiv.org/abs/2605.24709) narrow the proposed tangent-checkpoint mechanism. No compute-fair advantage over refresh-amortized replay was demonstrated. Do not conflate forward state reconstruction with gradient truncation.

Reversible abstraction: UCAgg already retains ground counts, balances within-block uncertainty, reconstructs aggregation and plans optimistically in the aggregate model. Source: https://infotech.unileoben.ac.at/fileadmin/shares/unileoben/lehrstuehle_department_institute/infotech/personal-sites/ronald_ortner/publikationen/AdAgg.pdf . C-UCRL is another direct statistical-sharing comparator: https://arxiv.org/abs/1910.04077 . Irreversible pooling is an insufficient baseline.

## Reproduction and artifacts

Files are adjacent in the session-local artifact directory:
- quantile-cycle-protocol.md: design frozen before execution, including correction of nonunique quantile boundaries before any run.
- quantile_cycle.py: deterministic NumPy search.
- quantile-cycle-result.json: reproduced search output and witness.
- certify_quantile_cycle.py: exact rational certificate, standard library only.
- quantile-cycle-certificate.json: exact orbit, strict gap and contraction information.
- quantile_control_baseline.py: same-world scalar-control comparator.
- quantile-control-baseline-result.json: executed comparator result.

Run the search and comparator with NumPy installed; the launch used the existing Yoked Plasticity virtual environment without modifying that project. Run `python3 certify_quantile_cycle.py` after placing the result JSON beside it. No RL training occurred and no public successor repository was created.

## Research boundary

The supported deliverable is the exact counterexample and a candidate theorem-level thesis. Larger quantile counts, damped/stochastic quantile updates, learned critics, robustness to MDP perturbations, prevalence, and practical impact remain unmeasured. Do not extrapolate the exact-backup orbit to QR-DQN training or claim that this simple witness establishes those extensions. A paper-level project requires those extensions or a genuinely new general characterization; merely adding the known scalar head would not clear the bar.
