# Quantile Cycles

**Finite midpoint-quantile control can have no fixed point—even with a unique optimal policy and no ties on its attracting orbit.**

This repository gives an explicit discounted MDP family and exact rational certificates for the ordinary **mean-greedy, hard quantile-projected Bellman control operator**. It is a theorem artifact, not a new learning algorithm or a practical QR-DQN failure claim.

Author: **mottopanikeiku** · License: **MIT** · Status: research note, not externally peer-reviewed; priority unestablished.

## Result

For every integer **K>=2**, construct a **one-state, two-action** MDP with rewards in **[0,1]** and discount `0 < beta <= 1/(2K)`:

- The projected control operator has an exact **two-cycle**, with unique quantiles and strictly alternating greedy actions.
- One action is uniquely optimal for expected return. The other cycle phase selects the genuinely suboptimal action.
- Every finite initial atom table converges to the cycle **modulo phase**, regardless of transient greedy tie choices. Consequently, the operator has **no fixed point**.
- For any prescribed **0<gamma<1**, a deterministic delay chain gives a finite-state example with exact primitive period **2L**, where `L = ceil(log(2K)/(-log gamma))`. This embedded cycle is locally attracting and its operator also has no fixed point.
- **K=1 is different:** the canonical single-atom operator is a contraction for discount below one, although its fixed point need not optimize expected return.

Read the [complete construction and proof](THEOREM.md). The quantifier is **for each K, a corresponding MDP**, not one fixed MDP failing at every capacity. The reward distribution depends on K, has K+1 outcomes, and the bad-policy loss shrinks with K. Holding the primitive discount fixed increases the number of states.

## Smallest explicit example

Use K=2, discount 1/4, and two self-looping actions:

| Action | Reward law |
|---|---|
| A | Always 91/128 |
| B | 0, 1/2, 1 with probabilities 3/16, 1/2, 5/16 |

The exact projected critic alternates between:

| Phase | A atoms | B atoms | Greedy action |
|---|---|---|---|
| Q_A | (107/120, 61/60) | (1307/1920, 2267/1920) | A |
| Q_B | (1793/1920, 1853/1920) | (347/480, 587/480) | B |

Both approximate greedy gaps are **3/128**. A's true mean-reward advantage is **19/128**; freezing the B policy loses **19/96** in discounted value. Backup closure is rational equality, not a tolerance-based recurrence detector.

## Reproduce

The current theorem checkers use only the Python standard library. Python 3.11+ is recommended. Run from the repository root:

```sh
python -B verify.py
```

This re-executes the checkers and compares their complete JSON certificates, including source hashes. Only elapsed time and Python version are excluded from equality. It checks:

- **18 one-state instances:** K=2 through 16, then 32, 64, 128;
- **5 fixed-discount instances:** gamma=9/10 and K=2,3,8,16,32;
- **18 zero-start trajectories:** six exact updates each, including the analytic translation formula.

The fixed-discount instances have **14,18,27,33,40 states** and fundamental periods **28,36,54,66,80**, respectively. Every primitive state backup is checked, rather than only treating the delay chain as a macro-step.

Individual outputs:

```sh
python -B certify.py
python -B zero_initialization.py
```

Finite certificates corroborate the proof. They do **not** prove the all-K or all-initializations statements by enumeration.

### Preserved earlier experiments

The earlier three-state K=3 search, exact six-cycle, damped period-26 certificate, robustness bounds, and sampled tabular continuation are retained under [`historical/`](historical/README.md). The sampled continuation **failed its larger-critic practical gate**. Those results are not evidence that this new analytic family causes sampled or deep-learning failure.

To verify the preserved artifacts as well:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-historical.txt
.venv/bin/python -B verify.py --historical
```

This reruns historical certificates, checks stored file hashes and deterministic reproduction, and recomputes sampled endpoints from policy traces. It does **not** retrain the sampled experiments; the historical page provides that separate command.

## Reading map

- [`THEOREM.md`](THEOREM.md): final theorem, proof, K=1 boundary, global attraction, fixed-discount embedding, and limits.
- [`PRIOR_ART.md`](PRIOR_ART.md): closest established results and what this construction does—and does not—add.
- [`HANDOFF.md`](HANDOFF.md): claim ledger, research graph, completed loops, and prerequisites for further work.
- [`results/`](results/): source-bound exact certificates and repository verification output.
- [`construction.md`](construction.md): frozen initial derivation used by the certificate hash. It is not the final theorem; the final note incorporates review clarifications and later global-attraction reasoning.
- [`historical/quantile-continuation/REPORT.md`](historical/quantile-continuation/REPORT.md): the earlier frozen practical gate and its negative outcome.

## Scope and attribution

Known work already establishes finite-quantile mean errors, strictly suboptimal projected choices for arbitrary K, and distributional control cycles involving optimal-action ties. The candidate contribution is the **specific conjunction** of arbitrary K>=2, strict suboptimal cycling, unique orbit quantiles, one-state global attraction, and fixed-discount embedding. A targeted primary-source review did not identify that exact conjunction; **absence from that search is not a priority proof**.

The operator here computes exact hard midpoint quantiles and then greedifies by their arithmetic mean. It is not a risk-sensitive quantile objective, Huber update, stochastic approximation algorithm, or deep network. No practical algorithmic advantage, capacity-independent loss, or external peer-review acceptance is claimed.
