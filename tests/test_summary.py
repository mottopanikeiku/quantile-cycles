"""Small fixtures test analysis, not the actual learning experiment."""
import copy
import json
import unittest
from pathlib import Path

from tools.summarize_sampled import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "sampled-config.json").read_text())
        self.population = []
        self.sampled = []
        for case in self.config["cases"]:
            for method in self.config["population_methods"]:
                schedules = ["none"] if method == "hard" else self.config["schedules"]
                for schedule in schedules:
                    self.population.append(self.row(case["name"], method, schedule, None, 0.5 if method != "scalar" else 0.0))
            for method in self.config["sampled_methods"]:
                for schedule in self.config["schedules"]:
                    for seed in self.config["seeds"]:
                        self.sampled.append(self.row(case["name"], method, schedule, seed, 0.5 if method != "scalar" else 0.0))

    @staticmethod
    def row(case, method, schedule, seed, regret):
        return {"case": case, "method": method, "schedule": schedule, "seed": seed,
                "endpoint_normalized_regret": regret, "endpoint_absolute_regret": regret / 10,
                "endpoint_switching_rate": 0.1}

    def test_all_four_gate_cells_required(self):
        result = summarize(self.config, self.population, self.sampled)
        self.assertTrue(result["network_gate"]["pass"])
        self.assertEqual(len(result["sampled_cells"]), 36)
        self.assertEqual(len(result["network_gate"]["cells"]), 4)
        for row in self.population:
            if (row["case"], row["method"], row["schedule"]) == ("family-32", "huber", "decaying"):
                row["endpoint_normalized_regret"] = 0.0
        self.assertFalse(summarize(self.config, self.population, self.sampled)["network_gate"]["pass"])

    def test_pairing_subtracts_scalar_by_seed(self):
        for row in self.sampled:
            row["endpoint_normalized_regret"] = row["seed"] / 100 + (0.3 if row["method"] != "scalar" else 0.0)
        result = summarize(self.config, self.population, self.sampled)
        quantile = next(row for row in result["sampled_cells"] if row["method"] == "huber")
        self.assertAlmostEqual(quantile["paired_normalized_excess_mean"], 0.3)
        for endpoint in quantile["paired_normalized_excess_ci95"]:
            self.assertAlmostEqual(endpoint, 0.3)

    def test_missing_seed_and_missing_cell_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.config, self.population, self.sampled[:-1])
        missing_cell = [row for row in self.sampled if (row["case"], row["method"], row["schedule"]) != ("family-32", "huber", "decaying")]
        with self.assertRaises(ValueError):
            summarize(self.config, self.population, missing_cell)

    def test_duplicate_seed_rejected(self):
        duplicate = copy.deepcopy(self.sampled)
        duplicate[0]["seed"] = duplicate[1]["seed"]
        with self.assertRaises(ValueError):
            summarize(self.config, self.population, duplicate)

    def test_population_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.config, self.population + self.population[:1], self.sampled)


if __name__ == "__main__":
    unittest.main()
