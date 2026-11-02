# Prior work

| Source | What it shows | How this differs |
| --- | --- | --- |
| [Rowland et al. 2019](https://proceedings.mlr.press/v97/rowland19a.html), appendix B.4 | For every finite K, midpoint-quantile means can strictly prefer a worse action. | Their misranking happens once; here it recurs in an attracting two-cycle. |
| [Bellemare, Dabney, Munos 2017](https://proceedings.mlr.press/v70/bellemare17a.html), proposition 2 | Exact distributional control can have a two-cycle and no fixed point. | The cycle comes from how ties between equal-mean actions are broken. Here every greedy choice is strict and one phase is truly suboptimal. |
| [Bellemare, Dabney, Rowland 2023](https://www.distributional-rl.org/contents/chapter7), example 7.11 | A one-state bounded-reward control cycle. | Relies on a custom tie selector among mean-optimal actions. |
| [Dabney et al. 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11791), proposition 2 | Midpoint quantile representation; projected evaluation contracts for a fixed policy. | The proof here uses that contraction inside each phase. |
| [Rowland et al. 2024](https://www.jmlr.org/papers/v25/23-0154.html) | Stochastic quantile TD converges for policy evaluation. | Evaluation, not control. |
| [Kuang et al. 2023](https://proceedings.mlr.press/v202/kuang23a.html) | Better scalar value estimates from a quantile critic. | A way to reduce the bias this example exploits. |

Rowland et al.'s example compares a reward of 1 with probability 1/(4K) against a deterministic 1/(8K). All K midpoint quantiles of the first action are zero, so the mean read from the quantiles prefers the deterministic action. That already gives strict suboptimality for every K. This repo adds feedback: the projection error enters the continuation distribution and produces a cycle.

Fixed-policy projected updates are contractions, but switching between contractive branches need not contract. Convergence results for projected control also assume mean preservation, which midpoint-quantile projection doesn't have. Capacity-dependent error bounds stay consistent with this example: its reward and policy gaps shrink like 1/K.
