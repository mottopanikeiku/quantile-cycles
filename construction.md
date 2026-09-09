# Candidate arbitrary-capacity construction, frozen before execution

This is a theoretical continuation, not a retuning of the failed practical experiment. No claim of novelty or practical relevance is authorized by this draft. All prior evidence remains unchanged.

## One-state family

Let integer K>=2, beta=1/(2K), epsilon=1/(4K^2), r_i=i/K (i=0..K), m=(K+1)/(2K), and S=(K-1)/(2K). Define c=m-beta*(1+beta)*S/2. One nonterminal state has two self-looping actions:

- A: deterministic reward c.
- B: reward r_0=0 with probability 1/(2K)-epsilon; reward r_i (i=1..K-1) with probability 1/K each; reward r_K=1 with probability 1/(2K)+epsilon.

Discount is beta. Each critic action has K equally weighted quantile atoms at midpoint levels tau_i=(i-1/2)/K. Select next actions by the arithmetic mean of their atoms, then apply exact quantile-projected Bellman backup.

The true mean reward of B is 1/2+epsilon. A's true mean reward is c. Claim: c>1/2+epsilon for all K>=2, so A is the unique true optimal stationary action.

## Projection identity on the proposed orbit

If selected continuation atoms x_1<...<x_K have beta*(x_K-x_1)<1/K, the target reward blocks r_j+beta*x_l do not overlap. Before reward block i (1..K), total probability is tau_i-epsilon. Since epsilon<w_i/K, the tau_i quantile is its first atom. Therefore the B backup is exactly

    B'_i = r_i + beta*x_1, i=1..K.

A's deterministic backup is A'_i=c+beta*x_i. In both phases below, selected continuation width is at most (K-1)/K, so beta<=1/(2K) guarantees the block separation.

## Closed-form two-cycle

Write s_i=r_i-m, i=1..K. Thus mean(s)=0 and s_1=-S.

Define M_B=(m+beta*c-beta^2*S)/(1-beta^2), M_A=c+beta*M_B.

Full action-critic phase Q_A (generated from selected B distribution M_B+s):

    Q_A(A,i)=c+beta*(M_B+s_i)
    Q_A(B,i)=r_i+beta*(M_B-S).

Full action-critic phase Q_B (generated from selected A distribution M_A+beta*s):

    Q_B(A,i)=c+beta*(M_A+beta*s_i)
    Q_B(B,i)=r_i+beta*(M_A-beta*S).

Conjectured identities: F(Q_A)=Q_B, F(Q_B)=Q_A. Q_A strictly selects A; Q_B strictly selects B. Both approximate greedy gaps equal g=beta*(1-beta)*S/2>0. Thus the second phase is strictly suboptimal for the real risk-neutral MDP. Every quantile is unique because tau_i lies strictly inside the first target atom's CDF jump.

Local attraction follows from phasewise fixed-policy beta contraction, with radius less than g/2 (choose g/4). The phases are distinct because their greedy actions differ. This is an exact orbit existence claim, not convergence from every initialization.

## K=1 boundary

With a single atom per action, selection by its mean is ordinary max over scalar atom values; all maximizing actions have the same one-atom continuation law. The hard projected operator is the lower median of R+gamma*max_a Q(S',a). Coupling and quantile nonexpansiveness make it gamma-Lipschitz in sup norm. Hence it has no nontrivial value cycle. This does not make its fixed point optimal for expected return.

## Fixed-discount extension candidate

Fix any gamma0 in (0,1). Choose integer L>=1 such that beta=gamma0^L<=1/(2K), and use exactly the formulas above with this beta (the algebra should hold for every beta in (0,1/(2K)]).

Replace the self-loop by a deterministic delay chain: decision state 0 has A/B reward then enters forced state 1; forced states 1..L-1 have zero reward and one action, and the last returns to state 0. For L=1, decision actions immediately return to state 0. There are L states total. The effective decision-epoch discount is beta.

Under synchronous primitive-state Bellman iteration, decision critics satisfy D_{t+L}=F_beta(D_t). Define D_t=Q_A for t mod 2L in 0..L-1 and Q_B otherwise. For forced state j=1..L-1, define its sole action atoms at iteration t as gamma0^(L-j) times the selected atoms of D_{t-(L-j)}. This should give an exact 2L-cycle of the full primitive-state operator. Decision greedy actions remain strict; forced states have no action choice. The two frozen stationary policy values at state 0 are c/(1-beta) and (1/2+epsilon)/(1-beta).

The fixed-discount construction uses O(log K / -log gamma0) states. It is not a one-state construction at arbitrary fixed discount. B has K+1 reward outcomes. Gaps shrink with K; no capacity-independent practical failure is claimed.

## Frozen checks

Before results: exact rational one-state checks for K=2..16,32,64,128, with beta=1/(2K). Fixed-discount checks at gamma0=9/10 for K=2,3,8,16,32; minimal L meeting the bound. Check all reward probabilities and reward bounds, true mean ordering, midpoint uniqueness, exact two-phase closure and greedy gaps, and primitive delay-chain backups over the full 2L orbit. No model search or fitted parameters. Independent mathematical and primary-literature reviews run concurrently.
