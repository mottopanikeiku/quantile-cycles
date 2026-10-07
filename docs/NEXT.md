# Next: formal proof and sampled quantile learning

The exact hard-projection theorem and sampled optimization are separate
questions. I completed the population and 32-seed tabular study in
[SAMPLED_RESULTS.md](SAMPLED_RESULTS.md): standard κ=1 Huber failed every
required material-effect comparison, so I stopped before replay/network
training. The [separate Lean PR](https://github.com/mottopanikeiku/quantile-cycles/pull/2)
checks the specific rational K=2 MDP and cycle with no sorry or added axioms.
The general real-valued theorem below remains a formalization proposal.

## Lean 4: formalize the general theorem, not just the certificates

### Representation and statement

Use Lean 4 with a pinned Mathlib revision. Represent each action critic by a
sorted `Fin K → ℝ` vector, with a proof that `2 ≤ K`. Represent reward laws as
finite supports and positive weights summing to one. Define the generalized
inverse quantile by its finite cumulative-mass characterization. This avoids
requiring an infinite-measure library to prove a finite-support construction.
Do not replace real initial atoms or real discounts by rationals: the executable
certificates are rational, but Theorems A and B quantify over reals.

Model greedy selection as a relation that permits any maximizing action.
The global-attraction theorem must cover every trajectory satisfying that
relation, including transient ties, rather than one fixed tie-breaking rule.
Define phasewise convergence explicitly: after some finite time, the even and
odd subsequences converge to opposite cycle tables, with either orientation.

### Lemmas in dependency order

1. **Finite quantiles.** Existence, generalized-inverse characterization,
   translation and positive-scale equivariance, and sup-norm nonexpansiveness
   under a coupling of equally indexed target outcomes. Handle coincident
   target atoms; do not assume every input atom is strictly ordered.
2. **Valid MDP and true values.** Positive probabilities summing to one,
   rewards in `[0,1]`, the formula `μ_B = 1/2 + ε`, the positive reward gap,
   and the two stationary-policy Bellman equations. Deduce unique optimality
   of A and distinguish the optimal action gap from stationary-policy loss.
3. **Separated reward blocks.** From `β (max x − min x) < 1/K`, prove the
   cumulative mass before block i is `τ_i − ε` and the next jump contains
   `τ_i` strictly. Derive `B'_i = i/K + β min x` and `A'_i = c + β x_i`.
4. **Cycle algebra.** Prove the closed-form means and centered shapes,
   strict greedy gap `g > 0`, unique orbit quantiles, exact two-backup
   closure and distinct phases. Rational-function simplification needs
   explicit proofs that denominators are nonzero.
5. **Global attraction.** Prove the width recurrence and eventual permanent
   separation; prove repeated A selection must end, then strict alternation.
   Reduce the selected means to the scalar affine recurrence with factor
   `β²`, prove its convergence, and lift convergence to the full tables.
   Exclude fixed points by applying this result to a constant trajectory.
6. **Local attraction and fixed discount.** Formalize the strict-policy
   neighborhoods and quantile contraction. Prove existence of a positive
   integer L with `γ^L ≤ 1/(2K)` using geometric decay; define the least such
   L first, then prove the logarithm/ceiling expression. Check each primitive
   delay-state backup and show the action word `A^L B^L` has primitive period
   `2L`. Eliminate forced states at a hypothetical fixed point. Do not claim
   global attraction to one delay-chain orientation.
7. **Boundary and explicit trajectory.** As separate corollaries, prove the
   K=1 discounted contraction and the zero-start translation formula.

The difficult part is likely the finite-quantile interface and the
all-initializations convergence argument, not the two-phase algebra.
An **estimate, not measured work**, is 3,000–6,000 Lean lines and 4–8 weeks
for someone familiar with Mathlib, including review. Begin with one week to
prove the finite-quantile characterization and nonexpansiveness; revise that
estimate based on the actual library gaps. A proof of only K=2 is useful for
checking definitions but must not be presented as the general theorem.

Completion of the general theorem would mean Lean checks its actual quantified
statements with no `sorry`, extra mathematical axioms or assumed cycle
identities. The specific K=2 proof does not meet that general-theorem scope.
Keep Python certificates as independently executable examples. Use a pinned
Lean/Mathlib toolchain and cached dependencies; the specific proof was built
in ephemeral CPU containers rather than claiming local performance.

## Sampled QR-DQN: a bounded, falsifiable experiment

### Question and scope

Does the strict cycle survive replacement of hard projection by quantile-loss
updates, especially the standard quantile Huber loss? The
[earlier sampled experiment](../historical/quantile-continuation/REPORT.md)
on a different witness failed its larger-critic practical criterion. That
negative result stays negative. This plan uses the new analytic family and
must report its own negative results without retuning the MDP to rescue them.

1. **Separate smoothing from sampling.** First compute expected pinball and
   quantile-Huber updates by enumerating the finite target law. Compare
   hard backups, damped hard backups, population-gradient updates and sampled
   updates. Standard Huber loss can change the population minimizer; a
   failure to cycle there is not an implementation failure. Validate loss
   gradients against finite differences away from nondifferentiable points.
2. **Start with tabular control.** Use the one-state family at K=2, 8 and 32,
   with `β=1/(2K)` for each. This is a different MDP at each K, not a capacity
   sweep on one environment. Include a separately labeled fixed-MDP sweep
   using the K=2 reward law and discount, while varying critic capacity.
   Evaluate both actions with a generative sampler before adding exploration.
3. **Fix the experiment before running.** Use 32 seeds, zero initialization,
   100,000 updates, 32 sampled reward/continuation pairs per action per update,
   pinball and Huber `κ=1`, and per-atom step schedules `0.05` and
   `0.2/(1+t/1000)^0.6`. Include scalar Q-learning with paired reward draws
   and the same schedules. Save code, configuration and seeds before the run;
   do not select schedules based on outcomes. Check sample counts against
   the K-dependent CDF margins and report uncertainty, not assumed resolution.
4. **Measure policy quality, not just switching.** Record critic means,
   greedy actions, exact stationary-policy regret, Bellman residuals and
   losses at fixed checkpoints. Primary endpoint: mean regret over the last
   20% of updates, normalized by the known always-B loss. Report absolute
   regret, paired excess regret versus scalar control, all seeds and paired
   95% bootstrap intervals. Report switching rate separately: noisy switches
   do not certify a periodic orbit. A bounded criterion worth testing is
   normalized excess regret ≥0.25 with a positive interval lower endpoint
   for Huber at K=8 and K=32 under both schedules. Report every cell regardless
   of that criterion; finite runs cannot prove asymptotic nonconvergence.
5. **Then test QR-DQN mechanisms.** Only if Huber retains a material effect,
   introduce a small CPU network on a one-hot state, uniform replay, minibatch
   32, Adam learning rate `0.001`, replay capacity 10,000, 1,000 warm-up steps,
   epsilon-greedy exploration decreasing from 1 to 0.05 over 10,000 steps,
   and target-network refresh every 100 steps. Fix 100,000 environment steps
   and 32 seeds; pair exploration/reward streams with a scalar DQN baseline.
   Repeat refresh intervals 1 and 1,000 as labeled mechanism checks, not
   outcome-selected settings. A one-state network adds no meaningful
   generalization test: describe it as a synthetic QR-DQN experiment, not
   evidence about deep RL generally. A later fixed-discount delay-chain run
   would test delayed bootstrapping separately.

The population and sampled tabular stages are now complete, including all
pre-specified settings, schedules and seeds. The
[compute record](../results/sampled-compute.json) gives the CPU-only cloud
cost, including both pilots. I did not reduce the number of seeds or select
a favorable schedule. The Huber material-effect rule failed in both stages,
so there is no replay/network result to report. The exact hard-projection
theorem remains valid; a future general Lean proof or a different scientific
question must not be described as rescuing this failed practical criterion.
