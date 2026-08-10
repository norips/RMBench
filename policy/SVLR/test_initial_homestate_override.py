import unittest

from policy.SVLR.runtime_config import apply_initial_homestate_override


class InitialHomestateOverrideTests(unittest.TestCase):
    @staticmethod
    def _args():
        return {
            "left_embodiment_config": {"homestate": [[0.0] * 7]},
            "right_embodiment_config": {"homestate": [[0.0] * 7]},
        }

    def test_override_updates_both_logical_arm_configs(self):
        args = self._args()
        expected = [float(index) for index in range(7)]

        apply_initial_homestate_override(args, expected)

        self.assertEqual(args["left_embodiment_config"]["homestate"], [expected])
        self.assertEqual(args["right_embodiment_config"]["homestate"], [expected])

    def test_string_override_is_parsed(self):
        args = self._args()

        apply_initial_homestate_override(args, "[0, 1, 2, 3, 4, 5, 6]")

        expected = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        self.assertEqual(args["left_embodiment_config"]["homestate"], expected)
        self.assertEqual(args["right_embodiment_config"]["homestate"], expected)

    def test_none_preserves_existing_homestates(self):
        args = self._args()
        before = {
            key: {"homestate": [row.copy() for row in value["homestate"]]}
            for key, value in args.items()
        }

        apply_initial_homestate_override(args, None)

        self.assertEqual(args, before)

    def test_override_rejects_wrong_joint_count(self):
        with self.assertRaisesRegex(ValueError, "expects 7"):
            apply_initial_homestate_override(self._args(), [0.0] * 6)

    def test_override_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite one-dimensional"):
            apply_initial_homestate_override(
                self._args(),
                [0.0, 1.0, 2.0, float("nan"), 4.0, 5.0, 6.0],
            )


if __name__ == "__main__":
    unittest.main()
