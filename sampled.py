"""Population and paired-sample tabular quantile control on the analytic family.

Updates use a frozen pre-update greedy critic, A wins mean ties, and each
atom takes a step on its own expected loss (there is no extra 1/K factor).
With u = target - atom, pinball's atom gradient is 1-tau for u<0,
-tau for u>0, and zero at u=0. Quantile Huber's atom gradient is
-|tau - 1[u<0]| * clip(u, -kappa, kappa) / kappa.

Sample streams are indexed by (seed, reward K, update, action, pair, stream),
not by method, schedule, case name, or critic capacity. Reward and continuation
uniforms use distinct streams. Each scalar update reuses the reward draws but
bootstraps its own pre-update maximum. No network experiment is implemented.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numba import __version__ as numba_version, njit


METHOD_CODES = {"hard": 0, "damped": 1, "pinball": 2, "huber": 3, "scalar": 4}


def family(k, beta=None):
    """Floating-point version of certify.family, including the true value gap."""
    if k < 2:
        raise ValueError("reward K must be at least two")
    beta = 1.0 / (2 * k) if beta is None else float(beta)
    if not 0 < beta <= 1.0 / (2 * k):
        raise ValueError("discount outside the analytic family's range")
    epsilon = 1.0 / (4 * k * k)
    midpoint = (k + 1.0) / (2 * k)
    spread = (k - 1.0) / (2 * k)
    c = midpoint - beta * (1 + beta) * spread / 2
    rewards = np.arange(k + 1, dtype=np.float64) / k
    weights = np.full(k + 1, 1.0 / k)
    weights[0] = 1.0 / (2 * k) - epsilon
    weights[-1] = 1.0 / (2 * k) + epsilon
    mean_b = 0.5 + epsilon
    m_b = (midpoint + beta * c - beta * beta * spread) / (1 - beta * beta)
    m_a = c + beta * m_b
    s = rewards[1:] - midpoint
    qa = np.array([c + beta * (m_b + s), rewards[1:] + beta * (m_b - spread)])
    qb = np.array([c + beta * (m_a + beta * s), rewards[1:] + beta * (m_a - beta * spread)])
    return {"reward_k": int(k), "beta": beta, "c": c, "rewards": rewards,
            "weights": weights, "mean_b": mean_b, "reward_gap": c - mean_b,
            "stationary_loss": (c - mean_b) / (1 - beta),
            "epsilon": epsilon, "cycle_gap": beta * (1 - beta) * spread / 2,
            "qa": qa, "qb": qb}


def midpoint_taus(k):
    return (np.arange(k, dtype=np.float64) + 0.5) / k


def midpoint_projection(values, weights, k):
    """Generalized inverse of a finite law; coincident outcomes are aggregated."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if k < 1 or values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("a nonempty finite law and positive capacity are required")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("law weights must be nonnegative and sum to one")
    output = np.empty(k)
    _project_general(values, weights, k, output)
    return output


def target_law(critic, model, action):
    """Exact finite target law, with a pre-update mean-greedy continuation."""
    critic = np.asarray(critic, dtype=np.float64)
    selected = int(critic[1].mean() > critic[0].mean())
    continuation = critic[selected]
    if action == 0:
        return model["c"] + model["beta"] * continuation, np.full(len(continuation), 1.0 / len(continuation))
    values = (model["rewards"][:, None] + model["beta"] * continuation).ravel()
    weights = np.repeat(model["weights"] / len(continuation), len(continuation))
    return values, weights


def hard_backup(critic, model):
    k = np.asarray(critic).shape[1]
    return np.array([midpoint_projection(*target_law(critic, model, a), k) for a in range(2)])


@njit(cache=True)
def _bounds(values, value, right):
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < value or (right and values[mid] == value):
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _target_bound(continuation, theta, reward, beta, right):
    lo, hi = 0, len(continuation)
    while lo < hi:
        mid = (lo + hi) // 2
        target = reward + beta * continuation[mid]
        if target < theta or (right and target == theta):
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _block_gradient(theta, tau, continuation, prefix, reward, beta, code, kappa):
    n = len(continuation)
    below = _target_bound(continuation, theta, reward, beta, False)
    above_start = _target_bound(continuation, theta, reward, beta, True)
    if code == 2:
        return ((1 - tau) * below - tau * (n - above_start)) / n
    low = _target_bound(continuation, theta - kappa, reward, beta, False)
    high = _target_bound(continuation, theta + kappa, reward, beta, True)
    negative = low * kappa + (below - low) * (theta - reward) - beta * (prefix[below] - prefix[low])
    positive = (n - high) * kappa + (high - above_start) * (reward - theta) + beta * (prefix[high] - prefix[above_start])
    return ((1 - tau) * negative - tau * positive) / (n * kappa)


def population_gradient(atoms, taus, continuation, rewards, weights, beta, loss="pinball", kappa=1.0):
    """Expected per-atom gradient; O(K * rewards * log(continuation K))."""
    if loss not in ("pinball", "huber") or kappa <= 0 or beta <= 0:
        raise ValueError("invalid loss, kappa, or discount")
    continuation = np.sort(np.asarray(continuation, dtype=np.float64))
    prefix = np.concatenate(([0.0], np.cumsum(continuation)))
    return np.array([sum(float(p) * _block_gradient(float(x), float(tau), continuation, prefix,
                                                  float(r), float(beta), METHOD_CODES[loss], float(kappa))
                         for r, p in zip(rewards, weights)) for x, tau in zip(atoms, taus)])


def expected_loss(atoms, taus, values, weights, loss="pinball", kappa=1.0):
    """Mean per-atom loss; targets are held constant when differentiating."""
    if loss not in ("pinball", "huber") or kappa <= 0:
        raise ValueError("invalid loss or kappa")
    total = 0.0
    values, weights = np.asarray(values), np.asarray(weights)
    for theta, tau in zip(atoms, taus):
        u = values - theta
        weight = np.where(u < 0, 1 - tau, tau)
        if loss == "pinball":
            penalty = np.abs(u)
        else:
            absolute = np.abs(u)
            penalty = np.where(absolute <= kappa, 0.5 * u * u, kappa * (absolute - 0.5 * kappa)) / kappa
        total += float(np.dot(weights, weight * penalty))
    return total / len(atoms)


@njit(cache=True)
def _mix64(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def _uniform(seed, reward_k, update, action, pair, stream):
    # SplitMix64 finalizer, with disjoint counters for both streams and actions.
    key = _mix64(np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15))
    key ^= _mix64(np.uint64(reward_k) + np.uint64(0xD1B54A32D192ED03))
    counter = np.uint64(update) * np.uint64(0x9E3779B97F4A7C15)
    counter += np.uint64(pair) * np.uint64(4) + np.uint64(action * 2 + stream)
    bits = _mix64(key ^ counter)
    return float(bits >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def paired_draws(seed, reward_k, update, batch_size=32):
    """Expose the two action-wise independent reward/continuation streams."""
    return np.array([[[_uniform(seed, reward_k, update, a, b, s) for s in range(2)]
                      for b in range(batch_size)] for a in range(2)])


@njit(cache=True)
def _project_general(values, masses, k, output):
    """Aggregate ties and compensate the cumulative sum before inverse lookup."""
    order = np.argsort(values, kind="mergesort")
    index, cumulative, correction = 0, 0.0, 0.0
    value = values[order[0]]
    for i in range(k):
        tau = (i + 0.5) / k
        while cumulative < tau and index < len(order):
            value = values[order[index]]
            while index < len(order) and values[order[index]] == value:
                increment = masses[order[index]] - correction
                advanced = cumulative + increment
                correction = (advanced - cumulative) - increment
                cumulative = advanced
                index += 1
            if cumulative >= tau or index == len(order):
                break
        output[i] = value


@njit(cache=True)
def _project_sorted(continuation, rewards, weights, beta, k, output):
    """Fast separated-block projection, general enumeration if blocks overlap."""
    separated = True
    width = beta * (continuation[-1] - continuation[0])
    for j in range(1, len(rewards)):
        if rewards[j] - rewards[j - 1] <= width:
            separated = False
    if separated:
        j, previous, cumulative = 0, 0.0, weights[0]
        for i in range(k):
            tau = (i + 0.5) / k
            while cumulative < tau and j < len(rewards) - 1:
                previous = cumulative
                j += 1
                cumulative += weights[j]
            # Search the actual CDF rather than rounding a conditional rank.
            lo, hi = 0, len(continuation) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if previous + weights[j] * (mid + 1) / len(continuation) >= tau:
                    hi = mid
                else:
                    lo = mid + 1
            output[i] = rewards[j] + beta * continuation[lo]
        return
    n = len(continuation) * len(rewards)
    values, masses = np.empty(n), np.empty(n)
    for j in range(len(rewards)):
        for l in range(len(continuation)):
            index = j * len(continuation) + l
            values[index] = rewards[j] + beta * continuation[l]
            masses[index] = weights[j] / len(continuation)
    _project_general(values, masses, k, output)


@njit(cache=True)
def _run_kernel(c, beta, rewards, weights, reward_k, k, codes, initials, scales, exponents,
                updates, batch_size, endpoint_start, checkpoints, seed, sampled, kappa):
    count = len(codes)
    q = np.zeros((count, 2, k))
    saved = np.zeros((len(checkpoints), count, 2, k))
    b_counts = np.zeros(count, dtype=np.int64)
    switch_counts = np.zeros(count, dtype=np.int64)
    all_switches = np.zeros(count, dtype=np.int64)
    mean_sums = np.zeros((count, 2))
    previous_actions = np.zeros(count, dtype=np.int64)
    continuation = np.empty(k)
    prefix = np.empty(k + 1)
    projected = np.empty(k)
    next_q = np.empty((2, k))
    reward_samples = np.empty((2, batch_size))
    uniforms = np.empty((2, batch_size))
    target_samples = np.empty(batch_size)
    reward_cdf = np.cumsum(weights)
    mean_b_reward = 0.0
    for j in range(len(rewards)):
        mean_b_reward += rewards[j] * weights[j]
    cp = 1  # validated checkpoints always start at zero
    for t in range(1, updates + 1):
        if sampled:
            for a in range(2):
                for b in range(batch_size):
                    u = _uniform(seed, reward_k, t, a, b, 0)
                    index = _bounds(reward_cdf, u, True)
                    if index == len(rewards):
                        index -= 1
                    reward_samples[a, b] = c if a == 0 else rewards[index]
                    uniforms[a, b] = _uniform(seed, reward_k, t, a, b, 1)
        for m in range(count):
            code = codes[m]
            selected = 0 if code == 4 else int(np.mean(q[m, 1]) > np.mean(q[m, 0]))
            alpha = initials[m]
            if exponents[m] != 0.0:
                alpha /= (1 + (t - 1) / scales[m]) ** exponents[m]
            if code == 4:
                best = max(q[m, 0, 0], q[m, 1, 0])
                for a in range(2):
                    mean_reward = c if a == 0 else mean_b_reward
                    if sampled:
                        mean_reward = np.mean(reward_samples[a])
                    next_q[a, 0] = q[m, a, 0] + alpha * (mean_reward + beta * best - q[m, a, 0])
            else:
                for i in range(k):
                    continuation[i] = q[m, selected, i]
                # Sorting does not change the uniform continuation law. Sampled
                # paths use original atom indices to preserve the explicit draw.
                if not sampled:
                    continuation.sort()
                    prefix[0] = 0.0
                    for i in range(k):
                        prefix[i + 1] = prefix[i] + continuation[i]
                for a in range(2):
                    if code <= 1:
                        if a == 0:
                            for i in range(k):
                                projected[i] = c + beta * continuation[i]
                        else:
                            _project_sorted(continuation, rewards, weights, beta, k, projected)
                        for i in range(k):
                            next_q[a, i] = projected[i] if code == 0 else q[m, a, i] + alpha * (projected[i] - q[m, a, i])
                    else:
                        if sampled:
                            for b in range(batch_size):
                                atom = min(int(uniforms[a, b] * k), k - 1)
                                target_samples[b] = reward_samples[a, b] + beta * continuation[atom]
                        for i in range(k):
                            theta, tau = q[m, a, i], (i + 0.5) / k
                            gradient = 0.0
                            if sampled:
                                for b in range(batch_size):
                                    u = target_samples[b] - theta
                                    if code == 2:
                                        gradient += (1 - tau) if u < 0 else (-tau if u > 0 else 0.0)
                                    else:
                                        weight = (1 - tau) if u < 0 else tau
                                        gradient -= weight * min(kappa, max(-kappa, u)) / kappa
                                gradient /= batch_size
                            elif a == 0:
                                gradient = _block_gradient(theta, tau, continuation, prefix, c, beta, code, kappa)
                            else:
                                for j in range(len(rewards)):
                                    gradient += weights[j] * _block_gradient(theta, tau, continuation, prefix,
                                                                            rewards[j], beta, code, kappa)
                            next_q[a, i] = theta - alpha * gradient
            # Both actions land only after both used the frozen old critic.
            active_k = 1 if code == 4 else k
            for a in range(2):
                for i in range(active_k):
                    q[m, a, i] = next_q[a, i]
            if code == 4:
                means_a, means_b = q[m, 0, 0], q[m, 1, 0]
            else:
                means_a, means_b = np.mean(q[m, 0]), np.mean(q[m, 1])
            action = int(means_b > means_a)
            switched = int(action != previous_actions[m])
            all_switches[m] += switched
            if t >= endpoint_start:
                b_counts[m] += action
                if t > endpoint_start:
                    switch_counts[m] += switched
                mean_sums[m, 0] += means_a
                mean_sums[m, 1] += means_b
            previous_actions[m] = action
        if cp < len(checkpoints) and t == checkpoints[cp]:
            saved[cp] = q
            cp += 1
    return saved, b_counts, switch_counts, all_switches, mean_sums


def _binomial_cdf(n, p, end):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(max(0, min(n, end) + 1)))


def checkpoint_metrics(critic, model, batch_size, kappa=1.0):
    """Exact Bellman/loss diagnostics and finite-CDF sampling resolution."""
    critic = np.asarray(critic, dtype=np.float64)
    k = critic.shape[1]
    means = critic.mean(axis=1)
    selected = int(means[1] > means[0])
    mean_targets = np.array([model["c"], model["mean_b"]]) + model["beta"] * max(means)
    residual, losses, crossings = 0.0, {"pinball": [], "huber": []}, []
    for a in range(2):
        values, weights = target_law(critic, model, a)
        projection = midpoint_projection(values, weights, k)
        residual = max(residual, float(np.max(np.abs(projection - critic[a]))))
        for loss in losses:
            losses[loss].append(expected_loss(critic[a], midpoint_taus(k), values, weights, loss, kappa))
        action_crossings = []
        for tau, value in zip(midpoint_taus(k), projection):
            left = min(1.0, max(0.0, float(weights[values < value].sum())))
            right = min(1.0, max(0.0, float(weights[values <= value].sum())))
            rank = math.ceil(batch_size * float(tau))
            miss = 1 - _binomial_cdf(batch_size, left, rank - 1) + _binomial_cdf(batch_size, right, rank - 1)
            action_crossings.append({"tau": float(tau), "quantile": float(value), "cdf_left": left,
                                     "cdf_right": right, "left_margin": float(tau) - left,
                                     "right_margin": right - float(tau),
                                     "cdf_left_standard_error": math.sqrt(left * (1 - left) / batch_size),
                                     "cdf_right_standard_error": math.sqrt(right * (1 - right) / batch_size),
                                     "empirical_crossing_miss_probability": max(0.0, min(1.0, miss))})
        crossings.append(action_crossings)
    return {"critic": critic.tolist(), "critic_means": means.tolist(), "greedy_action": "B" if selected else "A",
            "absolute_regret": selected * model["stationary_loss"], "hard_bellman_residual": residual,
            "mean_bellman_residual": float(np.max(np.abs(mean_targets - means))),
            "expected_pinball_loss_by_action": losses["pinball"],
            "expected_huber_loss_by_action": losses["huber"], "cdf_crossings_by_action": crossings}


def _prepare(config, stage):
    updates, batch_size = int(config["updates"]), int(config["batch_size"])
    start = int(config.get("endpoint_start", max(1, updates - updates // 5 + 1)))
    if updates < 1 or batch_size < 1 or not 1 <= start <= updates:
        raise ValueError("invalid update, batch, or endpoint dimensions")
    if config.get("initialization", "zero") != "zero" or config.get("tie_breaking", "A") != "A":
        raise ValueError("only zero initialization and A tie-breaking are defined")
    checkpoints = sorted(set([0, updates] + [int(t) for t in config.get("checkpoints", [])]))
    if checkpoints[0] != 0 or checkpoints[-1] != updates:
        raise ValueError("checkpoints must be between zero and updates")
    kappa = float(config.get("huber_kappa", 1.0))
    if kappa <= 0:
        raise ValueError("Huber kappa must be positive")
    methods = config.get(stage + "_methods", ["hard", "damped", "pinball", "huber", "scalar"] if stage == "population" else ["pinball", "huber", "scalar"])
    variants = []
    schedules = config["schedules"]
    for method in methods:
        if method not in METHOD_CODES or (stage == "sampled" and method not in ("pinball", "huber", "scalar")):
            raise ValueError("unsupported method for stage: " + method)
        names = [None] if method == "hard" else list(schedules)
        for name in names:
            schedule = {"initial": 1.0, "decay_scale": 1.0, "exponent": 0.0} if name is None else schedules[name]
            initial, scale, exponent = float(schedule["initial"]), float(schedule["decay_scale"]), float(schedule["exponent"])
            if initial <= 0 or scale <= 0 or exponent < 0:
                raise ValueError("invalid step schedule")
            variants.append({"method": method, "schedule": "none" if name is None else name,
                             "initial": initial, "decay_scale": scale, "exponent": exponent})
    if not variants:
        raise ValueError("at least one method is required")
    return updates, batch_size, start, checkpoints, kappa, variants


def _run(config, seeds, stage):
    updates, batch_size, start, checkpoints, kappa, variants = _prepare(config, stage)
    codes = np.array([METHOD_CODES[v["method"]] for v in variants], dtype=np.int64)
    initials = np.array([v["initial"] for v in variants])
    scales = np.array([v["decay_scale"] for v in variants])
    exponents = np.array([v["exponent"] for v in variants])
    records = []
    for case in config["cases"]:
        k, reward_k = int(case["critic_k"]), int(case["reward_k"])
        if k < 1:
            raise ValueError("critic capacity must be positive")
        model = family(reward_k, case.get("beta"))
        for seed in seeds:
            saved, b_counts, switches, all_switches, mean_sums = _run_kernel(
                model["c"], model["beta"], model["rewards"], model["weights"], reward_k, k,
                codes, initials, scales, exponents, updates, batch_size, start,
                np.array(checkpoints, dtype=np.int64), int(seed or 0), stage == "sampled", kappa)
            for m, variant in enumerate(variants):
                endpoint_count = updates - start + 1
                fraction = int(b_counts[m]) / endpoint_count
                metrics = []
                for index, t in enumerate(checkpoints):
                    critic = saved[index, m]
                    if variant["method"] == "scalar":
                        critic = critic[:, :1]
                    metrics.append({"update": t, **checkpoint_metrics(critic, model, batch_size, kappa)})
                records.append({"case": case["name"], "reward_k": reward_k, "critic_k": k,
                                "method": variant["method"], "schedule": variant["schedule"],
                                "seed": None if stage == "population" else int(seed),
                                "endpoint_normalized_regret": fraction,
                                "endpoint_absolute_regret": fraction * model["stationary_loss"],
                                "endpoint_switching_rate": int(switches[m]) / max(1, endpoint_count - 1),
                                "endpoint": {"start_update": start, "end_update": updates,
                                             "policy_observations": endpoint_count, "b_selections": int(b_counts[m]),
                                             "b_selection_fraction": fraction, "normalized_regret": fraction,
                                             "absolute_regret": fraction * model["stationary_loss"],
                                             "switches": int(switches[m]), "transition_observations": endpoint_count - 1,
                                             "switching_rate": int(switches[m]) / max(1, endpoint_count - 1),
                                             "critic_means": (mean_sums[m] / endpoint_count).tolist(),
                                             "final_critic_means": metrics[-1]["critic_means"],
                                             "all_update_switches": int(all_switches[m]),
                                             "all_update_switching_rate": int(all_switches[m]) / updates},
                                "checkpoints": metrics})
    return {"schema_version": 1, "stage": stage, "config": config,
            "metadata": {"dtype": "float64", "rng": "counter-indexed SplitMix64 finalizer; 53-bit uniforms",
                         "pairing_key": ["seed", "reward_k", "update", "action", "pair", "stream"],
                         "step_index": "alpha(t-1) for update t=1..updates",
                         "gradient_scaling": "expected loss per atom; no extra 1/K",
                         "endpoint_policy": "post-update greedy, inclusive start/end; A wins ties",
                         "scalar_target": "batch mean reward + beta * own pre-update max Q",
                         "scalar_checkpoint_hard_residual": "K=1 hard quantile operator, not mean Bellman operator",
                         "checkpoints": checkpoints, "seeds": [] if stage == "population" else list(seeds),
                         "numpy_version": np.__version__, "numba_version": numba_version},
            "models": [{"case": c["name"], "reward_k": int(c["reward_k"]), "critic_k": int(c["critic_k"]),
                        **{key: value for key, value in family(int(c["reward_k"]), c.get("beta")).items()
                           if key not in ("qa", "qb", "weights", "rewards", "reward_k")},
                        "rewards": family(int(c["reward_k"]), c.get("beta"))["rewards"].tolist(),
                        "weights": family(int(c["reward_k"]), c.get("beta"))["weights"].tolist()}
                       for c in config["cases"]], "rows": records}


def run_population(config):
    return _run(config, [None], "population")


def run_sampled(config, seeds=None):
    seeds = list(config["seeds"] if seeds is None else seeds)
    if any(int(seed) < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    return _run(config, seeds, "sampled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["population", "sampled"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    args = parser.parse_args()
    if (args.seed_start is None) != (args.seed_count is None):
        parser.error("--seed-start and --seed-count must be supplied together")
    config = json.loads(args.config.read_text())
    seeds = None if args.seed_start is None else range(args.seed_start, args.seed_start + args.seed_count)
    result = run_population(config) if args.stage == "population" else run_sampled(config, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
