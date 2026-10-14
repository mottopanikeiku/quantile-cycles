"""Guards of the exact checker and the README's K=2 numbers; standard library only."""
from fractions import Fraction as F
import json
import math
from pathlib import Path
import unittest
from unittest import mock

import certify
import verify

ROOT = Path(__file__).resolve().parents[1]


class ProjectionTests(unittest.TestCase):
    def test_midpoint_quantiles_and_margin(self):
        atoms, margin = certify.projection([(F(1), F(1, 2)), (F(0), F(1, 2))], 2)
        self.assertEqual(atoms, [0, 1])
        self.assertEqual(margin, F(1, 4))

    def test_cdf_touching_a_midpoint_is_rejected(self):
        # The certificate relies on every quantile being unique: a CDF step
        # landing exactly on tau=1/4 must fail rather than pick a side.
        with self.assertRaises(AssertionError):
            certify.projection([(F(0), F(1, 4)), (F(1), F(3, 4))], 2)

    def test_unnormalized_law_is_rejected(self):
        with self.assertRaises(AssertionError):
            certify.projection([(F(0), F(1, 2)), (F(1), F(1, 4))], 2)

    def test_tied_greedy_means_are_rejected(self):
        with self.assertRaises(AssertionError):
            certify.selected([[F(0), F(1)], [F(1), F(0)]])


class FamilyTests(unittest.TestCase):
    def test_parameter_range_is_enforced(self):
        with self.assertRaises(AssertionError):
            certify.family(1, F(1, 2))
        with self.assertRaises(AssertionError):
            certify.family(2, F(1, 4) + F(1, 1000))
        with self.assertRaises(AssertionError):
            certify.family(2, F(0))

    def test_smallest_instance_matches_readme(self):
        f = certify.family(2, F(1, 4))
        self.assertEqual(f["c"], F(91, 128))
        self.assertEqual(f["rewards"], [0, F(1, 2), 1])
        self.assertEqual(f["weights"], [F(3, 16), F(1, 2), F(5, 16)])
        row = certify.one_state(2)
        self.assertEqual(row["discount"], "1/4")
        self.assertEqual(row["greedy_gap"], "3/128")
        self.assertEqual(row["true_reward_gap"], "19/128")
        self.assertEqual(row["frozen_bad_policy_loss"], "19/96")
        self.assertEqual(row["cdf_margin"], "1/16")

    def test_delay_length_matches_closed_form(self):
        # README: L = ceil(log(2K) / -log(gamma)), period 2L.
        rows = json.loads((ROOT / "results" / "certificates.json").read_text())["fixed_discount"]
        self.assertEqual([row["exact_period"] for row in rows], [28, 36, 54, 66, 80])
        for row in rows:
            k, gamma = row["k"], F(row["discount"])
            length = math.ceil(math.log(2 * k) / -math.log(gamma))
            self.assertEqual(row["states"], length)
            self.assertEqual(row["exact_period"], 2 * length)
            self.assertLessEqual(gamma ** length, F(1, 2 * k))
            self.assertGreater(gamma ** (length - 1), F(1, 2 * k))


class VerifierTests(unittest.TestCase):
    def test_compare_rejects_any_changed_field(self):
        expected = json.loads((ROOT / "results" / "zero-initialization-certificate.json").read_text())
        changed = json.loads(json.dumps(expected))
        changed["checks"][0]["shift_at_iteration_two"] = "0"
        with mock.patch.object(verify, "run_json", return_value=changed):
            with self.assertRaises(RuntimeError):
                verify.compare("zero_initialization.py", "results/zero-initialization-certificate.json")
        with mock.patch.object(verify, "run_json", return_value=expected):
            self.assertEqual(
                verify.compare("zero_initialization.py", "results/zero-initialization-certificate.json"),
                expected,
            )

    def test_compare_ignores_only_named_fields(self):
        expected = json.loads((ROOT / "results" / "certificates.json").read_text())
        changed = dict(expected, wall_seconds=-1.0, python="0.0")
        with mock.patch.object(verify, "run_json", return_value=changed):
            verify.compare("certify.py", "results/certificates.json", ignored=("wall_seconds", "python"))
        changed["source_sha256"] = "0" * 64
        with mock.patch.object(verify, "run_json", return_value=changed):
            with self.assertRaises(RuntimeError):
                verify.compare("certify.py", "results/certificates.json", ignored=("wall_seconds", "python"))


if __name__ == "__main__":
    unittest.main()
