# Quantile-Huber learning avoids the hard-projection failure

Quantile-Huber learning with κ=1 matched or beat paired scalar Q-learning in every tested setting. Population Huber had zero final-window regret throughout. None of the four larger-capacity Huber comparisons met the excess-regret threshold for the replay/network experiment, so I stopped at tabular learning.

## What I measured

I ran exact finite-law population updates followed by paired sampled tabular learning. Each sampled setting uses 32 seeds, zero initialization, 100,000 synchronous updates and batch 32 per action. Constant steps are 0.05; decaying steps are `0.2/(1+t/1000)^0.6`, with t starting at zero. The [protocol](SAMPLED_PROTOCOL.md) and [configuration](../sampled-config.json) describe the [original study settings](https://github.com/mottopanikeiku/quantile-cycles/commit/beb7921).

The primary endpoint is mean stationary-policy regret over post-update policies 80,001 through 100,000, divided by the always-B loss. It equals the B-selection fraction. I average across seeds and subtract each paired scalar seed for excess regret; intervals use 10,000 seed-pair percentile-bootstrap resamples. Switching fractions count only transitions within that final window.

`family-K` uses the matching K-dependent reward law and discount 1/(2K). `fixed-K` uses the K=2 MDP at critic capacity K. Family-2 and fixed-2 are the same data under two labels.

## Population comparison

These are deterministic normalized endpoints from [the complete population run](../results/population.json.gz), also included in [the summary](../results/sampled-summary.json). Hard projection has no schedule; its endpoint is repeated in both rows only to make the comparison readable.

| Setting | Schedule | Hard | Damped hard | Population pinball | Population Huber | Scalar |
|---|---|---:|---:|---:|---:|---:|
| family-2 | constant | 0.50000 | 0.55175 | 0.40000 | 0.00000 | 0.00000 |
| family-2 | decaying | 0.50000 | 0.55455 | 0.40015 | 0.00000 | 0.00000 |
| family-8 | constant | 0.50000 | 0.50980 | 0.12500 | 0.00000 | 0.00000 |
| family-8 | decaying | 0.50000 | 0.51465 | 0.12510 | 0.00000 | 0.00000 |
| family-32 | constant | 0.50000 | 0.50000 | 0.11480 | 0.00000 | 0.00000 |
| family-32 | decaying | 0.50000 | 0.50250 | 0.04055 | 0.00000 | 0.00000 |
| fixed-2 | constant | 0.50000 | 0.55175 | 0.40000 | 0.00000 | 0.00000 |
| fixed-2 | decaying | 0.50000 | 0.55455 | 0.40015 | 0.00000 | 0.00000 |
| fixed-8 | constant | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| fixed-8 | decaying | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| fixed-32 | constant | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| fixed-32 | decaying | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

Huber selects A throughout the final window in every population setting and schedule. Pinball still chooses B on some updates, at a different frequency from the hard backup's exact half-time B policy. Damping hard projection leaves positive regret on the matching family. On the fixed K=2 MDP, increasing capacity to 8 or 32 gives zero final-window regret for every method.

## Sampled results

[sampled-summary.json](../results/sampled-summary.json) contains every seed, regret interval and paired difference at full precision. The [compressed raw file](../results/sampled-raw.json.gz) contains checkpoint critics, means, policies, expected losses, hard-projected and mean Bellman residuals, and CDF-resolution summaries. Atom-wise CDF diagnostics can be regenerated with `sampled.checkpoint_metrics`.

| Setting | Loss | Schedule | Normalized regret | Absolute regret | Paired normalized excess [95% CI] | Switching fraction |
|---|---|---|---:|---:|---:|---:|
| family-2 | huber | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-2 | huber | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-2 | pinball | constant | 0.375192 | 0.0742568 | +0.375192 [+0.373967, +0.376356] | 0.407809 |
| family-2 | pinball | decaying | 0.399388 | 0.0790454 | +0.399388 [+0.398341, +0.400400] | 0.394134 |
| family-2 | scalar | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-2 | scalar | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-32 | huber | constant | 0.004292 | 0.0000503 | -0.073964 [-0.076622, -0.071266] | 0.002411 |
| family-32 | huber | decaying | 0.000000 | 0.0000000 | -0.002286 [-0.002900, -0.001708] | 0.000000 |
| family-32 | pinball | constant | 0.103255 | 0.0012102 | +0.024998 [+0.023762, +0.026238] | 0.073299 |
| family-32 | pinball | decaying | 0.028464 | 0.0003336 | +0.026178 [+0.022469, +0.029897] | 0.015291 |
| family-32 | scalar | constant | 0.078256 | 0.0009172 | +0.000000 [+0.000000, +0.000000] | 0.037367 |
| family-32 | scalar | decaying | 0.002286 | 0.0000268 | +0.000000 [+0.000000, +0.000000] | 0.001180 |
| family-8 | huber | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-8 | huber | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-8 | pinball | constant | 0.079159 | 0.0037209 | +0.079159 [+0.075316, +0.082900] | 0.088593 |
| family-8 | pinball | decaying | 0.116367 | 0.0054699 | +0.116367 [+0.111072, +0.121698] | 0.086078 |
| family-8 | scalar | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| family-8 | scalar | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-2 | huber | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-2 | huber | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-2 | pinball | constant | 0.375192 | 0.0742568 | +0.375192 [+0.373967, +0.376356] | 0.407809 |
| fixed-2 | pinball | decaying | 0.399388 | 0.0790454 | +0.399388 [+0.398341, +0.400400] | 0.394134 |
| fixed-2 | scalar | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-2 | scalar | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-32 | huber | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-32 | huber | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-32 | pinball | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-32 | pinball | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-32 | scalar | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-32 | scalar | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-8 | huber | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-8 | huber | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-8 | pinball | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-8 | pinball | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-8 | scalar | constant | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| fixed-8 | scalar | decaying | 0.000000 | 0.0000000 | +0.000000 [+0.000000, +0.000000] | 0.000000 |

The four larger-capacity Huber comparisons have population excess regret zero. Sampled Huber also has zero regret at K=8 under both schedules. At K=32 with constant steps its normalized regret is 0.004292, versus scalar control's 0.078256; paired excess is −0.073964 [−0.076622, −0.071266]. With decaying steps Huber has zero final-window regret, versus scalar's 0.002286. The stopping rule required excess regret of at least 0.25 and a positive interval lower endpoint in every comparison; none met it.

Sampled pinball has positive paired excess in the matching family. At K=32 its paired excess is 0.024998 (constant) and 0.026178 (decaying), also below the threshold. All sampled losses and scalar control have zero observed final-window regret on fixed-8 and fixed-32.

## Resolving the quantile jump

The hard cycle's CDF margins are 1/(4K²). At the central midpoint, the batch-32 binomial CDF standard errors are about 0.07655, 0.08770 and 0.08835 for K=2, 8 and 32. The margins are only 0.81650, 0.04454 and 0.00276 standard errors, respectively. A normal-approximation calculation gives 185, 61,958 and 16,097,104 independent fixed-target samples to put that margin 1.96 standard errors away. This scale estimate does not account for drifting SGD targets. Exact values and assumptions are in [the summary](../results/sampled-summary.json).

## Reproduce

With Python 3.11 and a Modal account:

```sh
python -m pip install -r requirements-sampled.txt modal
modal run modal_sampled.py --stage population --output results/population.json.gz
modal run modal_sampled.py --stage sampled --output results/sampled-raw.json.gz
python -B tools/summarize_sampled.py
```

The local equivalents are `sampled.py --stage population|sampled --config sampled-config.json --output FILE.json`. They use the same kernels; the Modal runner compresses results and summarizes per-atom CDF rows before returning them. Pilot commands are `--stage population-pilot` and `--stage pilot`. I used CPU-only containers, 2 requested cores and 1 GiB each, with at most four containers. Library, OS and thread metadata are stored in each raw batch; the CPU model was unavailable. The [estimated cost upper bound](../results/sampled-compute.json) for all four jobs is about $0.022, including pilots and startup.

## Interpretation

- This is a synthetic one-state study with three family MDPs and a fixed-MDP capacity comparison.
- The endpoint evaluates each greedy policy as stationary; it does not measure the return of the changing training-time policy.
- Hard projection, expected gradients and sampled SGD are different operators. Finite regret and noisy switches do not establish asymptotic cycling or convergence.
- Zero bootstrap intervals occur when all 32 observed seed endpoints are zero; they do not exclude rare unseen seeds.
- The generative sampler removes exploration and replay. The network experiment was conditional on the Huber excess-regret result and was not run.
