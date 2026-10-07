# Sampled quantile learning: smoothing removes the material failure

I ran the analytic construction without changing its rewards after seeing the results. The exact hard projection keeps its strict two-cycle, but the standard κ=1 quantile-Huber loss does **not** preserve the pre-specified material excess-regret effect. All four Huber comparisons required to proceed to replay/network training fail the stopping rule. I therefore did not run the network stage.

## What I measured

I first ran the exact finite-law population updates, then the paired sampled tabular study. Each sampled cell uses 32 seeds, zero initialization, 100,000 synchronous updates and batch 32 per action. Constant steps are 0.05; decaying steps are `0.2/(1+t/1000)^0.6`, with t starting at zero. The [protocol](SAMPLED_PROTOCOL.md) and [configuration](../sampled-config.json) were committed in [beb7921](https://github.com/mottopanikeiku/quantile-cycles/commit/beb7921) before either pilot.

The primary endpoint is the mean stationary-policy regret over post-update policies 80,001 through 100,000, divided by the known always-B loss. It equals the B-selection fraction. It is **not** the return of the changing training-time policy. I average across seeds and subtract each paired scalar seed for excess regret; intervals use 10,000 seed-pair percentile-bootstrap resamples. Switching fractions count only transitions within that final window.

`family-K` uses the matching K-dependent reward law and discount 1/(2K). `fixed-K` always uses the K=2 MDP at critic capacity K. Family-2 and fixed-2 are identical data under two labels, not independent evidence.

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

Huber selects A throughout the final window in every population setting and schedule. Pinball still chooses B on some updates, but its finite-step dynamics do not reproduce the hard backup's exact half-time B policy. Damping hard projection does not remove that population failure on the matching family. Increasing capacity on the fixed K=2 MDP removes the observed failure at capacities 8 and 32; this is why the family must not be described as one MDP failing at every capacity.

## Every sampled cell

Full precision, every seed, both regret intervals and paired differences are in [sampled-summary.json](../results/sampled-summary.json). The [compressed raw file](../results/sampled-raw.json.gz) saves all checkpoint critics, means, policies, expected losses, both hard-projected and mean Bellman residuals, and CDF-resolution summaries. Atom-wise CDF diagnostics can be regenerated from those critics with `sampled.checkpoint_metrics`.

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

The four required Huber cells have population excess regret zero. Sampled Huber also has zero regret at K=8 under both schedules. At K=32 with constant steps its normalized regret is 0.004292, versus scalar control's 0.078256; paired excess is −0.073964 [−0.076622, −0.071266]. With decaying steps Huber has zero final-window regret, versus scalar's 0.002286. These results reject the fixed criterion of excess at least 0.25 with a positive interval lower endpoint; they do not prove asymptotic convergence.

Sampled pinball has positive paired excess in the matching family, but noisy switching is not proof of a periodic orbit. At K=32 its paired excess is only 0.024998 (constant) and 0.026178 (decaying), well below the material-effect threshold. All sampled losses and scalar control have zero observed final-window regret on fixed-8 and fixed-32.

## Can batch 32 resolve the hard quantile jump?

The hard cycle's CDF margins are 1/(4K²). At the central midpoint, the batch-32 binomial CDF standard errors are about 0.07655, 0.08770 and 0.08835 for K=2, 8 and 32. The margins are only 0.81650, 0.04454 and 0.00276 standard errors, respectively. A normal-approximation scale calculation gives 185, 61,958 and 16,097,104 independent fixed-target samples to put that margin 1.96 standard errors away; these are diagnostics, **not** confidence guarantees or effective sample counts for drifting SGD targets. Exact values and assumptions are in [the summary](../results/sampled-summary.json).

## Reproduce

With Python 3.11 and a Modal account:

```sh
python -m pip install -r requirements-sampled.txt modal
modal run modal_sampled.py --stage population --output results/population.json.gz
modal run modal_sampled.py --stage sampled --output results/sampled-raw.json.gz
python -B tools/summarize_sampled.py
```

The local equivalents are `sampled.py --stage population|sampled --config sampled-config.json --output FILE.json`. They use the same kernels; the Modal runner compresses results and summarizes bulky per-atom CDF rows before returning them. Pilot commands are `--stage population-pilot` and `--stage pilot`. I used CPU-only containers, 2 requested cores and 1 GiB each, at most four containers. Modal hides the CPU model; returned library/OS/thread metadata is stored in each raw batch. The [four study jobs cost upper bound](../results/sampled-compute.json) is about $0.022, including pilots and startup; this is not an invoice or a speed comparison.

## Limits

- This is one synthetic one-state construction, with three different family MDPs and a fixed-MDP comparison, not evidence about deep RL generally.
- Expected gradients, sampled SGD and hard projection are different operators. I did not prove a Huber fixed-point or convergence theorem.
- Zero bootstrap intervals occur when all 32 observed seed endpoints are zero; they do not rule out rare unseen seeds.
- Evaluating both actions with a generative sampler removes exploration/replay issues. Those mechanisms were deliberately not tested after the Huber rule failed.
- Finite noisy switching and nonzero regret cannot certify asymptotic cycling. The exact theorem and earlier negative historical experiment retain their separate scopes.

Written with AI coding assistance.
