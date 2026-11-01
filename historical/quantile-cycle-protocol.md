# Search for a strict quantile-control cycle

The search tests whether exact finite-quantile Bellman control can produce a persistent strict greedy-policy cycle in a small stationary discounted MDP. Scalar Bellman optimality provides the comparison.

## Search settings

NumPy generator seed 418; 512 MDPs; 3 states, 2 actions and 2 outcomes per state-action pair with probabilities 0.4 and 0.6; integer rewards in [-3,3]; discount 0.9; 3 uniform midpoint quantiles; zero initialization; 600 synchronous iterations.

Candidate periods range from 2 through 8 in the last 64 iterations. A candidate needs value-array periodicity within 1e-8, distinct greedy policies across the period and every chosen action's mean gap above 1e-4. The search returns the first qualifying witness.

## Unique quantiles

Probabilities 0.4/0.6 make cumulative target masses multiples of 1/15, while the requested quantiles are odd multiples of 1/6. This avoids CDF boundaries at the requested quantiles. An earlier unrun design used equiprobable outcomes, which would have allowed nonunique quantile minimizers.

## Comparator

Exact expected-reward Bellman optimality iteration runs on the same MDP. Its greedy policy and true policy values show whether the cycle selects a worse policy. Scalar Q control, C51's mean-preserving projection when its support contains the targets, and Quantiled Expansion Mean are relevant existing alternatives.
