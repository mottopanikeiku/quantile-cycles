# Related work

Finite-quantile suboptimality and distributional control cycles both have precedents. The construction in [THEOREM.md](THEOREM.md) combines strict suboptimal cycling with global phasewise attraction for each finite K≥2.

| Source | Established result | Relation to this construction |
|---|---|---|
| [Rowland et al. (2019), Statistics and Samples in Distributional Reinforcement Learning, Appendix B.4](https://proceedings.mlr.press/v97/rowland19a.html) | For every finite K, midpoint-quantile mean estimates can strictly prefer an inferior action. | A one-step wrong choice; the present example is recurrent. |
| [Bellemare, Dabney and Munos (2017), A Distributional Perspective on Reinforcement Learning, Proposition 2](https://proceedings.mlr.press/v70/bellemare17a.html) | Exact distributional control can have a two-cycle and no fixed point. | Uses distribution-dependent resolution of optimal-action mean ties. The present projected cycle has strict greedy choices and a suboptimal phase. |
| [Bellemare, Dabney and Rowland (2023), Distributional Reinforcement Learning, Chapter 7, Example 7.11](https://www.distributional-rl.org/contents/chapter7) | A one-state bounded-reward distributional control cycle. | Uses a custom tie selector among mean-optimal actions. |
| [Dabney et al. (2018), Distributional Reinforcement Learning with Quantile Regression, Proposition 2](https://ojs.aaai.org/index.php/AAAI/article/view/11791) | Midpoint-quantile representation, atom-mean control and fixed-policy projected contraction. | The present proof uses that contraction within each phase. |
| [Rowland et al. (2024), An Analysis of Quantile Temporal-Difference Learning](https://www.jmlr.org/papers/v25/23-0154.html) | Stochastic quantile policy-evaluation convergence. | Studies a fixed policy; the present control map changes its greedy policy. |
| [Kuang et al. (2023), Variance Control for Distributional Reinforcement Learning](https://proceedings.mlr.press/v202/kuang23a.html) | Quantile-derived mean estimation and Quantiled Expansion Mean. | An existing way to improve scalar control estimates from a quantile critic. |

## Finite-quantile bias

Rowland et al.'s Appendix B.4 compares a Bernoulli reward with probability 1/(4K) of reward one against a deterministic reward 1/(8K). All K midpoint quantiles of the Bernoulli action are zero. The represented mean strictly prefers the deterministic action despite its lower true expectation.

That example already has arbitrary-K strict suboptimality, unique midpoint quantiles, bounded rewards and the deterministic-action-versus-mixture structure. Here the projection bias also feeds back through the continuation distribution and produces a cycle.

## Distributional cycles

The 2017 paper and the book show that distributional Bellman control need not converge. The book's one-state example has a zero-reward self-looping action and an equally likely -1/+1 terminating action. Both have true mean zero. A distribution-dependent greedy tie rule makes the represented distributions cycle.

The present construction has strict represented-mean choices in both phases, a unique true optimal action and recurring selection of its inferior alternative. Its cycle attracts every finite initial atom table modulo phase.

## Compatibility with convergence bounds

With a fixed selected policy, the projected quantile operator contracts in the maximal quantile sup metric. Switching between contractive branches need not produce a globally contractive control map.

Chapter 7's projected-control convergence results also impose mean preservation and other metric assumptions. Midpoint-quantile projection is not mean-preserving. Capacity-dependent approximation bounds remain consistent with this example: its reward and policy gaps shrink as 1/K.

## Reward bounds and fixed discounts

Rewards lie in [0,1], while their support grows to K+1 outcomes. The one-state construction requires β≤1/(2K). A standard deterministic delay chain keeps a prescribed primitive discount γ fixed by choosing β=γ^L. It adds L−1 forced states and changes the primitive cycle period to 2L.

Rational parameters give exact rational certificates. The theorem also permits real reward parameters for an arbitrary real discount.

## Scope

The [sampled study](docs/SAMPLED_RESULTS.md) found that Huber smoothing removed the material excess-regret effect on this family. The [earlier fixed example](historical/README.md) also failed to produce the larger-critic sampled effect being tested.
