# Quantile-control discriminator

Question: can exact finite-quantile projected Bellman optimality iteration exhibit a persistent strict greedy-policy cycle in a tiny stationary discounted MDP, while scalar Bellman optimality converges? This is not a claim of novelty or an RL training run. It isolates projection/control feedback without sampling, optimization, replay, or neural approximation.

Frozen before execution: NumPy generator seed 418; 512 MDPs; 3 states, 2 actions, 2 outcomes with probabilities 0.4 and 0.6 per action; integer rewards in [-3,3]; discount 0.9; 3 uniform midpoint quantiles; zero initialization; 600 synchronous iterations. Search periods 2 through 8 in the last 64 iterations. A candidate must have value-array periodicity within 1e-8, distinct greedy policies across the period, and every chosen action's mean gap above 1e-4. Retain first candidate, or report no witness in this finite search. No parameter search after observing failure.

Pre-execution correction: original draft used equiprobable outcomes, placing target CDF boundaries exactly on the requested quantiles and allowing nonunique quantile minimizers. Probabilities 0.4/0.6 avoid that artifact: cumulative atom probabilities are multiples of 1/15, while quantile fractions are odd multiples of 1/6. No experiment ran under the draft.

Comparator: exact expected-reward Bellman optimality iteration on the same MDP; its greedy policy and true policy values identify whether cycle is consequential. A positive witness justifies further primary-literature checking, not a new algorithm claim. Scalar Q control, C51's mean-preserving projection (when support contains targets), and Quantiled Expansion Mean are mandatory collisions/baselines. A mean-preserving scalar correction alone is not presumed new.
