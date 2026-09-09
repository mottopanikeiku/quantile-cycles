# Prior art and claim boundary

This is a targeted primary-source comparison, not a systematic-review protocol or proof of priority. No exact match was located for the conjunction in `THEOREM.md`. An unindexed paper, thesis, technical report, or folklore construction could still contain it. Automated literature review is not external peer review.

## Closest results

| Source | Already established | Difference from the present construction |
|---|---|---|
| [Rowland et al. (2019), Statistics and Samples in Distributional Reinforcement Learning, Appendix B.4](https://proceedings.mlr.press/v97/rowland19a.html) | For every finite K, midpoint-quantile mean estimates can strictly prefer a truly inferior action. | A one-step/terminal wrong choice, not a recurrent strict cycle. |
| [Bellemare, Dabney and Munos (2017), A Distributional Perspective on Reinforcement Learning, Proposition 2](https://proceedings.mlr.press/v70/bellemare17a.html) | Exact distributional control can have a two-cycle and no fixed point. | Uses distribution-dependent resolution of optimal-action mean ties; no finite quantile projection and no strictly suboptimal phase. |
| [Bellemare, Dabney and Rowland (2023), Distributional Reinforcement Learning, Chapter 7, Example 7.11](https://www.distributional-rl.org/contents/chapter7) | One-state, bounded-reward cyclical distributional control. | Again relies on a custom tie selector among mean-optimal actions rather than unique represented-mean argmaxes. |
| [Dabney et al. (2018), Distributional Reinforcement Learning with Quantile Regression, Proposition 2](https://ojs.aaai.org/index.php/AAAI/article/view/11791) | The relevant midpoint-quantile representation, atom-mean control convention, and fixed-policy projected contraction. | The contraction is for a fixed policy, not the changing greedy control composition. |
| [Rowland et al. (2024), An Analysis of Quantile Temporal-Difference Learning](https://www.jmlr.org/papers/v25/23-0154.html) | Stochastic quantile policy-evaluation convergence under stated assumptions. | Does not prove convergence of hard mean-greedy control. |
| [Kuang et al. (2023), Variance Control for Distributional Reinforcement Learning](https://proceedings.mlr.press/v202/kuang23a.html) | Quantile-derived mean estimation and QEM. | A relevant existing mitigation, not a new algorithm proposed by this repository. |

### The arbitrary-K suboptimality precedent is very close

Rowland et al.'s Appendix B.4 compares a Bernoulli reward with probability 1/(4K) of reward one against a deterministic reward 1/(8K). All K midpoint quantiles of the Bernoulli action are zero. The represented mean therefore strictly prefers the deterministic action, despite its lower true expectation.

That result already owns the finite-quantile mean-bias mechanism, arbitrary-K strict suboptimality, unique midpoint quantiles, bounded rewards, and the broad deterministic-action-versus-mixture motif. None of those ingredients alone is a novelty claim here.

### The distributional-cycle precedent is also close

The 2017 paper and the book already show that distributional Bellman control need not converge. The book's one-state example has a zero-reward self-looping action and an equally likely -1/+1 terminating action. Both have true mean zero. A specially chosen legal greedy tie rule makes the represented distributions cycle.

Thus one state, period two, bounded rewards, or distributional nonconvergence are not separating properties by themselves. The present cycle instead has strict represented-mean choices at both phases, a unique true optimal action, and recurring selection of its inferior alternative.

### Why positive convergence results do not rule this out

With a fixed selected policy, the projected quantile operator contracts in the maximal quantile sup metric. The present proof uses precisely that contraction within each phase. Switching between different contractive branches need not produce a globally contractive control map.

Chapter 7's projected-control convergence results also impose mean preservation and other metric assumptions. Midpoint-quantile projection is not mean-preserving. The theorem therefore does not contradict those results.

Approximation-error bounds that decrease with K do not ensure preservation of an argmax whose action gap also decreases with K. Here the reward and policy gaps are of order 1/K. This is not a counterexample to capacity-dependent approximation bounds.

## Candidate contribution, stated narrowly

The candidate contribution is an explicit closed-form construction combining:

1. every finite midpoint-quantile capacity K>=2;
2. a one-state, continuing, two-action MDP;
3. ordinary hard greedy selection by atom means;
4. an exact two-cycle with strict action switches and unique orbit quantiles;
5. a unique true optimal policy, with its inferior alternative selected every other update;
6. global phasewise attraction from every finite initial atom table;
7. a standard delay embedding giving a no-fixed-point example at every fixed 0<gamma<1.

The local cycle construction received a targeted literature collision check. The later global-attraction proof received separate independent automated algebra review; the absence of a matching local construction is not stronger evidence of global-priority status.

## Normalization and embedding are not separate novelty claims

The final reward law is uniformly bounded in [0,1], with K+1 distinct reward outcomes. Reward **range** is bounded independently of K; support **cardinality** grows with K. An earlier unnormalized derivation had a growing reward range. That is not the final theorem's parameterization.

The direct one-state construction requires beta<=1/(2K). A deterministic delay chain keeps a prescribed primitive discount gamma fixed by choosing beta=gamma^L. This is a standard effective-discount transformation, not a new technique. It adds L-1 forced states and changes the primitive cycle period to 2L.

For rational parameters the executable certificates use rational arithmetic. The theorem at an arbitrary real gamma permits real reward parameters; it does not claim rational data for every real discount.

## What would establish more

External specialist review and a deeper priority search are prerequisites for a first-result or publication-significance claim. Any practical claim would need a separately frozen experiment on the new family. The failed practical gate on the older witness remains negative evidence and cannot be reset by changing the theorem.
