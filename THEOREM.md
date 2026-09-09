# Strict cycles in finite quantile control

Author: mottopanikeiku. Research note; not externally peer-reviewed. Exact executable checks and independent automated algebra review accompany, but do not replace, the proof below. Priority is not claimed.

## 1. Operator and quantifiers

A state-action return law is represented by K equally weighted real atoms. The control rule selects an action by their arithmetic mean. A synchronous backup forms the exact law of R + gamma times a uniformly selected atom of the greedy next action, then projects to midpoint quantiles tau_i=(i-1/2)/K, i=1,...,K. Quantiles use the canonical generalized inverse. Greedy ties may be resolved arbitrarily.

This note concerns that hard projected Bellman control operator, not a quantile-risk objective, stochastic quantile SGD, Huber regression, target networks, or deep QR-DQN.

**Theorem A.** For every integer K>=2 and every 0<beta<=1/(2K), there is a one-state, two-action discounted MDP with rewards in [0,1] whose K-midpoint-quantile mean-greedy control operator has a two-cycle. Both phases have unique quantiles and strict greedy choices; one phase selects the unique truly suboptimal action. The two-cycle is globally attracting modulo phase from every finite real initial atom table, under arbitrary transient greedy tie choices. In particular, the operator has no fixed point.

**Theorem B.** For every fixed discount 0<gamma<1 and every integer K>=2, there is a finite MDP with bounded rewards in [0,1], a unique optimal policy, and an exact locally attracting cycle of fundamental period 2L, where

    L = ceil(log(2K)/(-log gamma)).

It has L states: one two-action decision state and L-1 forced states with one legal action each. Its projected control operator has no fixed point.

**Boundary.** At K=1, for any finite discounted MDP with bounded rewards and 0<=gamma<1, the canonical one-atom projected operator is a gamma-contraction and has no nontrivial value cycle. Its fixed point need not be optimal for expected return.

The quantifier order matters: **for every K, a corresponding MDP is constructed**. This is not one fixed MDP that fails at all capacities. In Theorem A the discount may shrink with K; Theorem B keeps the discount fixed by increasing the number of states. Neither asserts capacity-independent loss or practical learning failure.

## 2. One-state construction

Fix K>=2 and 0<beta<=1/(2K). Put

    epsilon = 1/(4K^2)
    r_i = i/K,                         i=0,...,K
    m = (K+1)/(2K)
    S = (K-1)/(2K)
    c = m - beta(1+beta)S/2.

Both actions return to the only state. Action A pays the deterministic reward c. Action B has rewards r_0,...,r_K and probabilities

    p_0 = 1/(2K) - epsilon
    p_i = 1/K,                         1<=i<K
    p_K = 1/(2K) + epsilon.

All probabilities are positive and sum to one. Action B's true mean reward is mu_B=1/2+epsilon. Define

    delta = c - mu_B.

It decreases with beta, so substitution of beta=1/(2K) gives

    delta >= (6K^2 - 3K + 1)/(16K^3) > 1/(4K) > 0.

Consequently 0<mu_B<c<m<1, every reward lies in [0,1], and A is the unique optimal action. The two frozen stationary-policy values differ by delta/(1-beta). The true optimal Q-value action gap is delta, not delta/(1-beta).

## 3. Quantile projection identity

Let x_1<=...<=x_K be the selected continuation atoms. Suppose

    beta(x_K-x_1) < 1/K.

The reward blocks r_j+beta*x_l are then separated. Before reward block i, for 1<=i<=K, the cumulative target probability is

    sum_{j<i} p_j = (i-1/2)/K - epsilon = tau_i - epsilon.

The first continuation atom within block i has probability p_i/K>epsilon. Therefore tau_i lies strictly inside that atom's CDF jump and the unique projected B atom is

    B'_i = r_i + beta*x_1.

Coincident minimum continuation atoms only enlarge this jump. No strict ordering of all input atoms is needed. The deterministic A backup is

    A'_i = c + beta*x_i.

## 4. Explicit two-cycle

Write s_i=r_i-m for i=1,...,K. Then mean(s)=0 and min(s)=-S. Set

    M_B = (m + beta*c - beta^2*S)/(1-beta^2)
    M_A = c + beta*M_B.

Define the full action-critic table Q_A by

    Q_A(A,i) = c + beta(M_B+s_i)
    Q_A(B,i) = r_i + beta(M_B-S),

and Q_B by

    Q_B(A,i) = c + beta(M_A+beta*s_i)
    Q_B(B,i) = r_i + beta(M_A-beta*S).

The selected distribution at Q_A is M_A+beta*s; the selected distribution at Q_B is M_B+s. Both satisfy the separation condition in Section 3, because their widths are at most (K-1)/K and beta<=1/(2K).

Let

    g = beta(1-beta)S/2 > 0.

The action means at Q_A are (M_A, M_A-g), while those at Q_B are (M_B-g, M_B). Thus Q_A strictly selects A and Q_B strictly selects B. Substitution into the two backup identities gives exactly

    F(Q_A) = Q_B,       F(Q_B) = Q_A.

Since their selected actions differ, the tables are distinct. Every projected quantile is unique. Q_B's greedy policy is genuinely suboptimal by Section 2.

### Local attraction

Within a sup-norm radius rho<g/2 of either phase, the greedy action remains unchanged: each action mean moves by at most rho. With a fixed selected policy, couple each reward/continuation target to its unperturbed counterpart. Every target moves by at most beta times the input sup-norm error; quantile projection has the same bound. The phase balls map into one another with contraction factor beta, so the two-step map contracts by beta^2. One may use rho=g/4.

This proof does not require local affine source routing or strict ordering of every target value.

## 5. Global attraction from finite initial tables

Let W_t be the maximum within-action atom range at iteration t. A target support has width at most 1+beta*W_t, so

    W_{t+1} <= 1 + beta*W_t,
    limsup beta*W_t <= beta/(1-beta) <= 1/(2K-1) < 1/K.

Hence after finitely many iterations the block-separation condition holds permanently. From one iteration later, every B critic has centered shape s by Section 3.

For a selected continuation distribution x, write h=mean(x)-min(x)>=0 and d=m-c=beta(1+beta)S/2. In the separated regime the next action-mean difference is

    mean(A') - mean(B') = -d + beta*h.

If B is selected, h=S, so the next table selects A by gap g and its selected shape is beta*s. At the following update h=beta*S, so B wins by gap g. Strict alternation is then permanent.

If B were never selected in this regime, repeated A updates would give h_{t+1}=beta*h_t. Eventually -d+beta*h_t<0, forcing B. A transient equality tie that chooses A can delay this switch, but cannot prevent it.

After the first selected B in the permanently separated regime, the centered shapes therefore alternate between s and beta*s. The remaining degree of freedom is their mean. If the B-selected mean is h, then the next A-selected mean is c+beta*h and the subsequent B-selected mean is

    h_next = m + beta*c - beta^2*S + beta^2*h.

This scalar recurrence contracts to M_B; the intervening mean contracts to M_A. The full critic tables converge phasewise to Q_A and Q_B. The phase orientation may depend on initialization and tie choices. This proves global attraction modulo phase, not finite-time arrival at the exact critic values.

A fixed point or a different periodic orbit would contradict this convergence. In particular, F has no fixed point.

### Zero initialization, explicitly

At Q_0=0 the tied actions have identical zero continuation laws, so the first backup is independent of tie-breaking:

    Q_1(A,i)=c,       Q_1(B,i)=r_i.

It strictly selects B. For n>=2 the exact iterates satisfy

    Q_n = Q_A + beta^(n-1)(m-M_B) * 1,    n even,
    Q_n = Q_B + beta^(n-1)(m-M_B) * 1,    n odd.

The formula follows by computing Q_2 and using translation equivariance F(Q+a*1)=F(Q)+beta*a*1. It is verified exactly for the finite instance list in the companion certificate.

## 6. A smallest explicit example

Take K=2 and beta=1/4. Action A pays 91/128. Action B pays 0, 1/2, and 1 with probabilities 3/16, 1/2, and 5/16.

| Phase | A atoms | B atoms | Greedy action |
|---|---|---|---|
| Q_A | (107/120, 61/60) | (1307/1920, 2267/1920) | A |
| Q_B | (1793/1920, 1853/1920) | (347/480, 587/480) | B |

Both approximate greedy gaps are 3/128. The true mean-reward gap is 19/128; the frozen bad-policy value loss is 19/96. This is an exact two-state-of-the-iteration cycle in a **one-environment-state** MDP.

## 7. Fixed-discount delay embedding

Fix 0<gamma<1. Choose the minimal integer

    L = ceil(log(2K)/(-log gamma)),       beta=gamma^L<=1/(2K).

Use the preceding rewards and formulas with this beta. Decision state 0 pays the A/B reward and enters state 1. Each forced state j=1,...,L-1 has one zero-reward action leading to j+1, with the last returning to state 0. For L=1, decision actions directly self-loop.

Let D_t be the decision-state critic table at iteration t. Write sel(D) for its selected action's atom vector. For an orbit, the forced-state table can be expressed as

    C_{j,t} = gamma^(L-j) sel(D_{t-(L-j)}).

These tables satisfy every primitive forced-state backup, and the decision backup gives the delayed recurrence

    D_{t+L} = F_beta(D_t).

Define D_t to equal Q_A for residues 0,...,L-1 modulo 2L and Q_B for residues L,...,2L-1. Together with the C_{j,t} formula, this is an exact full-operator orbit. The decision word A^L B^L has fundamental period 2L, so the full orbit has exactly that period, not merely a divisor detected numerically.

Every decision choice is strict and every forced state has a single legal action. All quantiles are unique. The local phase-ball argument now contracts each primitive backup by gamma and the full return map by gamma^(2L). There are exactly L states, with the uniform bound

    L <= 1 + log(2K)/(-log gamma).

If a full primitive-operator fixed point existed, eliminating its forced states would give a fixed point of F_beta, contradicting Theorem A. Thus the fixed-discount example also has no fixed point.

At state 0 the frozen stationary-policy values are c/(1-beta) and (1/2+epsilon)/(1-beta). The forced states add no reward. No claim is made here that arbitrary initial full tables converge to this particular A^L B^L orientation; the embedding can carry different phase alignments across residue classes.

## 8. Why K=1 is different

For one atom per action, choosing by the atom mean is simply scalar maximization. Tied maximizing actions have the same singleton continuation law. The backed-up scalar is the lower median of

    R + gamma*max_a Q(S',a).

Couple identical rewards and next states between two tables Q and Q'. Their targets differ by at most gamma*||Q-Q'||_infinity. Quantile nonexpansiveness therefore gives

    ||F(Q)-F(Q')||_infinity <= gamma*||Q-Q'||_infinity.

For 0<=gamma<1, Banach's theorem yields a unique fixed point and excludes nontrivial value cycles. At gamma=1 this argument is not a strict contraction and the conclusion need not hold. Optimizing through median projection is not generally risk-neutral optimal.

## 9. Prior work and limits

Closest established results include:

- [Rowland et al., Statistics and Samples in Distributional Reinforcement Learning, Appendix B.4](https://proceedings.mlr.press/v97/rowland19a.html): strict suboptimal finite-quantile choices for arbitrary finite K, but a terminal/fixed-point construction rather than a recurrent cycle.
- [Bellemare, Dabney and Rowland, Distributional Reinforcement Learning, Chapter 7](https://www.distributional-rl.org/contents/chapter7): distributional nonconvergence involving ties among optimal actions; mean-preserving projections; existing scalar-control/distributional-evaluation separation.
- [Dabney et al., Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044): fixed-policy projected contraction, not contraction of the changing mean-greedy control operator.
- [Rowland et al., An Analysis of Quantile Temporal-Difference Learning](https://www.jmlr.org/papers/v25/23-0154.html): stochastic policy-evaluation convergence, not this hard control map.
- [Kuang et al., Variance Control for Distributional Reinforcement Learning](https://proceedings.mlr.press/v202/kuang23a.html): quantile-derived mean estimation and QEM, a relevant existing mitigation.

The new candidate statement is the conjunction of arbitrary finite K>=2, strict suboptimal cycling, unique orbit quantiles, one-state global attraction, and fixed-discount finite-state embedding. This note does not claim priority merely because a targeted review found no matching example.

The constructed loss is of order 1/K, not bounded below by a capacity-independent constant. The reward law itself depends on K and has K+1 outcomes; its probability offsets are of order 1/K^2. These facts are consistent with approximation-error bounds and make practical extrapolation particularly unsafe. Historical sampled experiments on a different K=3 witness failed their larger-critic practical gate and remain preserved without relabeling.

## 10. Executable evidence and review corrections

`certify.py` checks exact rational probabilities, reward bounds, true value equations, greedy gaps, quantile uniqueness, cycle closure and every primitive delay-chain backup. It checks K=2,...,16,32,64,128 for one state, and gamma=9/10 with K=2,3,8,16,32 for the delay construction. `zero_initialization.py` checks the exact zero-start formula through six updates for all one-state instances. Finite checks corroborate the symbolic proof; they are not the proof of the universal quantifiers.

The frozen pre-execution draft `construction.md` remains unchanged for source binding. The final proof makes two review-requested qualifications explicit: K=1 requires gamma<1, and the logarithmic state bound uses the minimal qualifying L. The global-attraction argument was derived after that initial draft and independently checked separately. Automated review is not external peer review.
