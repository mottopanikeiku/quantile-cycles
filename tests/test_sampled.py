"""Independent small references for sampled.py; no experiment data fixtures."""
import copy
from fractions import Fraction
import json
import unittest

import numpy as np

import certify
import sampled


def tiny_config(updates=9):
    return {"updates": updates, "batch_size": 4, "seeds": [0, 7],
            "endpoint_start": updates - 3, "initialization": "zero", "tie_breaking": "A",
            "huber_kappa": 1.0, "checkpoints": [0, 1, 3],
            "schedules": {"constant": {"initial": 0.05, "decay_scale": 1000., "exponent": 0.},
                          "decaying": {"initial": 0.2, "decay_scale": 1000., "exponent": 0.6}},
            "cases": [{"name": "family-2", "reward_k": 2, "critic_k": 2}],
            "population_methods": ["hard", "damped", "pinball", "huber", "scalar"],
            "sampled_methods": ["pinball", "huber", "scalar"]}


def brute_gradient(atoms, taus, values, weights, loss, kappa):
    answer = []
    for theta, tau in zip(atoms, taus):
        u = values - theta
        if loss == "pinball":
            gradient = np.where(u < 0, 1 - tau, np.where(u > 0, -tau, 0.0))
        else:
            gradient = -np.where(u < 0, 1 - tau, tau) * np.clip(u, -kappa, kappa) / kappa
        answer.append(np.dot(weights, gradient))
    return np.array(answer)


def reference_sampled(config, seed, method, schedule):
    """Slow direct target enumeration, keeping both pre-update actions frozen."""
    case = config["cases"][0]
    model = sampled.family(case["reward_k"])
    k = 1 if method == "scalar" else case["critic_k"]
    q = np.zeros((2, k))
    trajectories = [q.copy()]
    settings = config["schedules"][schedule]
    for t in range(1, config["updates"] + 1):
        draws = sampled.paired_draws(seed, case["reward_k"], t, config["batch_size"])
        selected = int(q[1].mean() > q[0].mean())
        alpha = settings["initial"] / (1 + (t - 1) / settings["decay_scale"]) ** settings["exponent"]
        next_q = np.empty_like(q)
        for a in range(2):
            rewards = np.full(config["batch_size"], model["c"]) if a == 0 else model["rewards"][
                np.searchsorted(np.cumsum(model["weights"]), draws[a, :, 0], side="right")]
            if method == "scalar":
                next_q[a] = q[a] + alpha * (rewards.mean() + model["beta"] * q.max() - q[a])
            else:
                indices = (draws[a, :, 1] * k).astype(int)
                targets = rewards + model["beta"] * q[selected, indices]
                gradient = brute_gradient(q[a], sampled.midpoint_taus(k), targets,
                                          np.full(len(targets), 1 / len(targets)), method, config["huber_kappa"])
                next_q[a] = q[a] - alpha * gradient
        q = next_q
        trajectories.append(q.copy())
    return trajectories


class FamilyTests(unittest.TestCase):
    def test_matches_exact_certificate_and_backups(self):
        for k in (2, 8, 32):
            exact = certify.family(k, Fraction(1, 2 * k))
            model = sampled.family(k)
            for key in ("c", "weights", "rewards", "qa", "qb"):
                np.testing.assert_allclose(model[key], np.asarray(exact[key], dtype=float), rtol=0, atol=2e-15)
            self.assertAlmostEqual(model["mean_b"], float(exact["mean_b"]), places=15)
            self.assertAlmostEqual(model["cycle_gap"], float(exact["gap"]), places=15)
            for phase in ("qa", "qb"):
                q, _ = certify.backup([exact[phase]], certify.make_model(exact, 1), Fraction(1, 2 * k), k)
                np.testing.assert_allclose(sampled.hard_backup(model[phase], model), np.asarray(q[0], dtype=float), atol=2e-15, rtol=0)

    def test_generalized_inverse_ties_and_cdf_boundaries(self):
        np.testing.assert_array_equal(sampled.midpoint_projection([0, 1, 1, 2], [.25] * 4, 2), [0, 1])
        np.testing.assert_array_equal(sampled.midpoint_projection([3, 3, 3], [.2, .3, .5], 8), [3] * 8)
        np.testing.assert_array_equal(sampled.midpoint_projection([99, 0, 2], [0, .5, .5], 2), [0, 2])

    def test_optimized_projection_overlap_and_separation(self):
        model = sampled.family(2)
        for continuation in (np.array([0., 0.5, 0.8]), np.array([-10., 0., 10.]), np.zeros(3)):
            for k in (2, 8, 32):
                output = np.empty(k)
                sampled._project_sorted(continuation, model["rewards"], model["weights"], model["beta"], k, output)
                values = (model["rewards"][:, None] + model["beta"] * continuation).ravel()
                weights = np.repeat(model["weights"] / len(continuation), len(continuation))
                np.testing.assert_allclose(output, sampled.midpoint_projection(values, weights, k), atol=1e-14, rtol=0)

    def test_scalar_reward_mean_and_true_gap(self):
        for k in (2, 8, 32):
            model = sampled.family(k)
            self.assertAlmostEqual(float(np.dot(model["rewards"], model["weights"])), model["mean_b"], places=15)
            self.assertGreater(model["reward_gap"], 1 / (4 * k))
            optimal_value = model["c"] / (1 - model["beta"])
            optimal_q_b = model["mean_b"] + model["beta"] * optimal_value
            self.assertAlmostEqual(optimal_value - optimal_q_b, model["reward_gap"], places=14)
            self.assertAlmostEqual(optimal_value - model["mean_b"] / (1 - model["beta"]), model["stationary_loss"], places=14)
        config = tiny_config(updates=1)
        config["endpoint_start"] = 1
        config["checkpoints"] = [0, 1]
        config["population_methods"] = ["scalar"]
        result = sampled.run_population(config)
        model = sampled.family(2)
        for row in result["rows"]:
            initial = config["schedules"][row["schedule"]]["initial"]
            np.testing.assert_allclose(row["checkpoints"][-1]["critic_means"], initial * np.array([model["c"], model["mean_b"]]), atol=1e-15)


class GradientTests(unittest.TestCase):
    def test_population_equals_brute_finite_law(self):
        continuation = np.array([-.8, .03, .03, 1.5])
        atoms = np.array([-1.2, .14, .91, 2.1])
        taus = sampled.midpoint_taus(4)
        rewards, weights, beta = np.array([0., .31, 1.]), np.array([.2, .5, .3]), .23
        values = (rewards[:, None] + beta * continuation).ravel()
        law_weights = np.repeat(weights / len(continuation), len(continuation))
        for loss in ("pinball", "huber"):
            for kappa in (.3, 1., 2.):
                actual = sampled.population_gradient(atoms, taus, continuation, rewards, weights, beta, loss, kappa)
                np.testing.assert_allclose(actual, brute_gradient(atoms, taus, values, law_weights, loss, kappa), atol=2e-15, rtol=0)

    def test_finite_difference_away_from_kinks(self):
        atoms, taus = np.array([-.8, .27, 1.2]), sampled.midpoint_taus(3)
        values, weights = np.array([-.3, .11, .75, 1.7]), np.array([.1, .2, .4, .3])
        for loss in ("pinball", "huber"):
            for kappa in (.4, 1.):
                expected = brute_gradient(atoms, taus, values, weights, loss, kappa)
                for i in range(len(atoms)):
                    plus, minus = atoms.copy(), atoms.copy()
                    plus[i] += 1e-6
                    minus[i] -= 1e-6
                    # expected_loss averages atoms; updates use per-atom gradients.
                    actual = len(atoms) * (sampled.expected_loss(plus, taus, values, weights, loss, kappa) -
                                           sampled.expected_loss(minus, taus, values, weights, loss, kappa)) / 2e-6
                    self.assertAlmostEqual(actual, expected[i], places=8)

    def test_pinball_exact_ties_are_zero(self):
        continuation, beta, reward = np.array([.123456789]), .23, .31
        theta = reward + beta * continuation[0]
        for tau in (.1, .5, .9):
            gradient = sampled.population_gradient([theta], [tau], continuation, [reward], [1.], beta)
            self.assertEqual(gradient[0], 0.)


class RunTests(unittest.TestCase):
    def test_rng_repeatability_and_separate_streams(self):
        draws = sampled.paired_draws(19, 2, 7, 32)
        np.testing.assert_array_equal(draws, sampled.paired_draws(19, 2, 7, 32))
        self.assertTrue(np.all((draws >= 0) & (draws < 1)))
        self.assertFalse(np.array_equal(draws[:, :, 0], draws[:, :, 1]))
        self.assertFalse(np.array_equal(draws, sampled.paired_draws(20, 2, 7, 32)))
        self.assertFalse(np.array_equal(draws[0], draws[1]))

    def test_sampled_reference_and_endpoint_accounting(self):
        config = tiny_config()
        result = sampled.run_sampled(config, [7])
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result["rows"]), 6)
        for row in result["rows"]:
            trajectory = reference_sampled(config, 7, row["method"], row["schedule"])
            self.assertEqual([p["update"] for p in row["checkpoints"]], [0, 1, 3, config["updates"]])
            for point in row["checkpoints"]:
                np.testing.assert_allclose(point["critic"], trajectory[point["update"]], atol=2e-15, rtol=0)
                for key in ("hard_bellman_residual", "expected_pinball_loss_by_action", "expected_huber_loss_by_action", "cdf_crossings_by_action"):
                    self.assertIn(key, point)
            actions = [int(q[1].mean() > q[0].mean()) for q in trajectory]
            endpoint = actions[config["endpoint_start"]:]
            self.assertEqual(row["endpoint"]["policy_observations"], len(endpoint))
            self.assertEqual(row["endpoint"]["transition_observations"], len(endpoint) - 1)
            self.assertEqual(row["endpoint"]["b_selections"], sum(endpoint))
            self.assertEqual(row["endpoint_normalized_regret"], sum(endpoint) / len(endpoint))
            self.assertEqual(row["endpoint"]["switches"], sum(a != b for a, b in zip(endpoint, endpoint[1:])))
            model = sampled.family(2)
            self.assertAlmostEqual(row["endpoint_absolute_regret"], row["endpoint_normalized_regret"] * model["stationary_loss"])
            means = np.mean([q.mean(axis=1) for q in trajectory[config["endpoint_start"]:]], axis=0)
            np.testing.assert_allclose(row["endpoint"]["critic_means"], means, atol=2e-15)

    def test_filtered_methods_case_names_and_capacities_preserve_pairing(self):
        config = tiny_config()
        full = sampled.run_sampled(config, [0])
        repeat = sampled.run_sampled(config, [0])
        self.assertEqual(full, repeat)
        filtered = copy.deepcopy(config)
        filtered["sampled_methods"] = ["huber"]
        filtered["cases"][0]["name"] = "fixed-2"
        part = sampled.run_sampled(filtered, [0])
        for row in part["rows"]:
            expected = next(r for r in full["rows"] if r["method"] == row["method"] and r["schedule"] == row["schedule"])
            self.assertEqual(row["checkpoints"], expected["checkpoints"])
            self.assertEqual(row["endpoint"], expected["endpoint"])
        filtered["sampled_methods"] = ["scalar"]
        filtered["cases"][0]["critic_k"] = 8
        for row in sampled.run_sampled(filtered, [0])["rows"]:
            expected = next(r for r in full["rows"] if r["method"] == "scalar" and r["schedule"] == row["schedule"])
            self.assertEqual(row["checkpoints"], expected["checkpoints"])
            self.assertEqual(row["endpoint_normalized_regret"], expected["endpoint_normalized_regret"])

    def test_population_updates_match_direct_frozen_target_reference(self):
        config = tiny_config(updates=4)
        config["checkpoints"] = list(range(5))
        config["cases"][0]["critic_k"] = 8
        result = sampled.run_population(config)
        model = sampled.family(2)
        for row in result["rows"]:
            k = 1 if row["method"] == "scalar" else 8
            q = np.zeros((2, k))
            for t in range(1, 5):
                alpha = 1.0
                if row["method"] != "hard":
                    settings = config["schedules"][row["schedule"]]
                    alpha = settings["initial"] / (1 + (t - 1) / settings["decay_scale"]) ** settings["exponent"]
                next_q = np.empty_like(q)
                for a in range(2):
                    values, weights = sampled.target_law(q, model, a)
                    if row["method"] == "scalar":
                        reward = model["c"] if a == 0 else model["mean_b"]
                        next_q[a] = q[a] + alpha * (reward + model["beta"] * q.max() - q[a])
                    elif row["method"] in ("hard", "damped"):
                        projection = sampled.midpoint_projection(values, weights, k)
                        next_q[a] = q[a] + alpha * (projection - q[a])
                    else:
                        gradient = brute_gradient(q[a], sampled.midpoint_taus(k), values, weights, row["method"], 1.0)
                        next_q[a] = q[a] - alpha * gradient
                q = next_q
                np.testing.assert_allclose(row["checkpoints"][t]["critic"], q, atol=3e-15, rtol=0)

    def test_population_hard_zero_start_exact_trajectory(self):
        config = tiny_config(updates=6)
        config["checkpoints"] = list(range(7))
        config["population_methods"] = ["hard"]
        for k in (2, 8, 32):
            config["cases"] = [{"name": "family", "reward_k": k, "critic_k": k}]
            row = sampled.run_population(config)["rows"][0]
            exact = certify.family(k, Fraction(1, 2 * k))
            q = [[[exact["c"]] * k, exact["rewards"][1:]]]
            for t in range(1, 7):
                if t > 1:
                    q, _ = certify.backup(q, certify.make_model(exact, 1), Fraction(1, 2 * k), k)
                np.testing.assert_allclose(row["checkpoints"][t]["critic"], np.asarray(q[0], dtype=float), atol=2e-15, rtol=0)


if __name__ == "__main__":
    unittest.main()
