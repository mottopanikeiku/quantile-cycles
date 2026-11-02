# Sampled learning on the cycle family

Does the hard-projection cycle show up when the critic is trained by gradient steps from samples? Settings are in [`sampled-config.json`](../sampled-config.json); the code is [`sampled.py`](../sampled.py).

## Setup

- MDPs: the one-state family from [THEOREM.md](../THEOREM.md) at K = 2, 8 and 32, with discount 1/(2K). A fourth set holds the K = 2 MDP fixed and widens the critic to 8 or 32 atoms ("fixed-K").
- Methods: hard projection, damped hard projection, expected (population) pinball and Huber gradients, and sampled pinball, Huber (κ = 1) and scalar Q-learning.
- Sampled runs: 32 seeds, 100,000 synchronous updates, 32 sampled reward/continuation pairs per action per update, zero initialization. The same random draws feed every loss, so the comparison with scalar Q-learning is paired.
- Step sizes: constant 0.05, or decaying 0.2/(1 + t/1000)^0.6.
- Next action: the arithmetic mean of the current critic's atoms; A wins exact ties.
- Score: the share of greedy choices that were the worse action B over updates 80,001 to 100,000. This equals stationary-policy regret divided by the always-B loss. Intervals are 10,000 percentile-bootstrap resamples over the 32 seed pairs.

The plan was to follow up with a replay-buffer network experiment only if Huber showed excess regret of at least 0.25 over scalar control in all four K = 8 and K = 32 comparisons. None got there, so that experiment doesn't exist.

## Results

Share of the last 20,000 greedy choices that picked B, averaged over seeds:

| MDP | Steps | Pinball | Huber | Scalar |
| --- | --- | ---: | ---: | ---: |
| K = 2 | constant | 37.5% | 0 | 0 |
| K = 2 | decaying | 39.9% | 0 | 0 |
| K = 8 | constant | 7.9% | 0 | 0 |
| K = 8 | decaying | 11.6% | 0 | 0 |
| K = 32 | constant | 10.3% | 0.43% | 7.8% |
| K = 32 | decaying | 2.8% | 0 | 0.23% |

Hard projection picks B half the time on every family MDP, as the cycle predicts. Population Huber and scalar learning pick it never. Population pinball sits in between (40% at K = 2, 12.5% at K = 8, 11% at K = 32 with constant steps).

At K = 32 with constant steps Huber beats scalar Q-learning by 7.4 points (95% interval 7.1 to 7.7). On the fixed K = 2 MDP with 8 or 32 atoms, every method has zero regret.

## Why sampling may hide the cycle

The hard cycle rests on a CDF jump of size 1/(4K²). Against the binomial noise of a 32-sample batch, that margin is 0.8 standard errors at K = 2, 0.04 at K = 8 and 0.003 at K = 32. Putting it 1.96 standard errors clear at K = 32 would take about 16 million independent samples per fixed target. That ignores target drift, so read it as a scale, not a bound. I haven't tested this explanation directly.

## Reproduce

```sh
pip install -r requirements-sampled.txt
python -B tools/summarize_sampled.py --check        # recompute the summary from saved runs
python sampled.py --stage sampled --config sampled-config.json --output out.json   # rerun locally; the full grid is the long step
```

`modal_sampled.py` splits the same runs over Modal CPU containers (about two cents in total, see [`results/sampled-compute.json`](../results/sampled-compute.json)). Seeds, checkpoints and intervals are in [`results/sampled-summary.json`](../results/sampled-summary.json); raw runs in `results/sampled-raw.json.gz` and `results/population.json.gz`.

Zero-width intervals mean all 32 seeds had zero regret; they don't rule out rare failures.
