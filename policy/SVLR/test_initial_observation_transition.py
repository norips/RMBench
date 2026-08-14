import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch, sentinel

import numpy as np

from envs.observe_and_pickup import observe_and_pickup
from policy.SVLR.deploy_policy import (
    DEFAULT_ENDPOSE,
    SimServer,
    extract_camera_payload,
)


class InitialObservationTransitionTests(unittest.TestCase):
    @staticmethod
    def _server():
        server = SimServer.__new__(SimServer)
        server.controlled_arm = "right"
        server.call_vlm_before_llm = True
        server.drive = True
        server.home_on_reset = False
        server.home_controlled = np.zeros(8)
        server.mirror_single_arm = False
        server.camera_key = "right_camera"
        server.vlm_camera_shader_dir = ""
        server.save_debug_images = False
        server.bridge = Mock()
        server.bridge.episode_id.return_value = 1
        server._episode_q = Mock()
        server._mark_driver_started = Mock()
        server._save_debug_camera_images = Mock()
        server._reset_svlr_episode = Mock()
        return server

    @staticmethod
    def _env(observation, with_hook=True):
        attributes = {
            "get_instruction": Mock(return_value="pick the matching object"),
            "get_obs": Mock(return_value=observation),
            "eval_success": False,
            "max_reward": 0.0,
            "take_action_cnt": 0,
        }
        if with_hook:
            attributes["on_initial_observation_published"] = Mock()
        return SimpleNamespace(**attributes)

    def test_transition_runs_after_snapshot_publish(self):
        events = []
        observation = {"observation": {}, "endpose": {}}
        env = self._env(observation)
        env.on_initial_observation_published.side_effect = lambda: events.append(
            "transition"
        )
        server = self._server()
        server.bridge.publish.side_effect = lambda *_args: events.append("publish")

        with (
            patch(
                "policy.SVLR.deploy_policy._endpose_from_obs",
                return_value=np.asarray(DEFAULT_ENDPOSE),
            ),
            patch(
                "policy.SVLR.deploy_policy.extract_camera_payload",
                return_value={"view": "initial"},
            ),
            patch(
                "policy.SVLR.deploy_policy.sim_list_entities",
                return_value=set(),
            ),
        ):
            server._home_and_engage(env, observation)

        self.assertEqual(events, ["publish", "transition"])
        server.bridge.publish.assert_called_once()
        server._episode_q.put.assert_called_once_with(
            (1, "pick the matching object")
        )
        server._mark_driver_started.assert_called_once_with(1)

    def test_environment_without_transition_hook_keeps_normal_flow(self):
        observation = {"observation": {}, "endpose": {}}
        env = self._env(observation, with_hook=False)
        server = self._server()

        with (
            patch(
                "policy.SVLR.deploy_policy._endpose_from_obs",
                return_value=np.asarray(DEFAULT_ENDPOSE),
            ),
            patch(
                "policy.SVLR.deploy_policy.extract_camera_payload",
                return_value={"view": "normal"},
            ),
            patch(
                "policy.SVLR.deploy_policy.sim_list_entities",
                return_value=set(),
            ),
        ):
            server._home_and_engage(env, observation)

        server.bridge.publish.assert_called_once()
        server._episode_q.put.assert_called_once_with(
            (1, "pick the matching object")
        )

    def test_environments_without_transition_hook_report_no_transition(self):
        self.assertFalse(SimServer._apply_initial_observation_transition(object()))

    def test_driver_runs_one_vlm_then_llm(self):
        server = self._server()
        client = Mock()

        server._run_svlr_episode(client, 1, "pick the matching object")

        self.assertEqual(
            client.predict.call_args_list,
            [
                call(api_name="/process_vlm"),
                call(
                    prompt="pick the matching object",
                    api_name="/process_llm_command",
                ),
            ],
        )
        server.bridge.open_action_window.assert_called_once_with(1)

    def test_restore_observation_pose_reuses_initial_view_command(self):
        server = self._server()
        initial = np.arange(16, dtype=np.float64)
        server._observation_cmd = initial.copy()
        restored_observation = {"frame": "restored"}
        server._execute_until_ee_reached = Mock(
            return_value=(1, restored_observation)
        )

        result = server._restore_observation_pose(sentinel.env)

        self.assertEqual(result, restored_observation)
        np.testing.assert_array_equal(server._cmd, initial)
        server._execute_until_ee_reached.assert_called_once_with(sentinel.env)

    def test_initial_view_remembers_command_instead_of_tracking_offset(self):
        observation = {"observation": {}, "endpose": {}}
        env = self._env(observation)
        server = self._server()
        server.home_on_reset = True
        server.home_controlled = np.asarray(
            [0.0, -0.15, 1.4, 0.5, -0.5, 0.5, 0.5, 1.0]
        )
        measured_before = np.arange(16, dtype=np.float64)
        measured_after = measured_before + 0.125
        server._execute_until_ee_reached = Mock(return_value=(1, observation))

        with (
            patch(
                "policy.SVLR.deploy_policy._endpose_from_obs",
                side_effect=[measured_before, measured_after],
            ),
            patch(
                "policy.SVLR.deploy_policy.extract_camera_payload",
                return_value={"view": "initial"},
            ),
            patch(
                "policy.SVLR.deploy_policy.sim_list_entities",
                return_value=set(),
            ),
        ):
            server._home_and_engage(env, observation)

        expected = measured_before.copy()
        expected[8:16] = server.home_controlled
        np.testing.assert_array_equal(server._observation_cmd, expected)
        self.assertFalse(np.array_equal(server._observation_cmd, measured_after))

    def test_empty_vlm_shader_reuses_synchronized_rgbd_capture(self):
        observation = {
            "observation": {
                "right_camera": {
                    "rgb": np.full((8, 10, 3), 127, dtype=np.uint8),
                    "depth": np.full((8, 10), 800.0, dtype=np.float32),
                }
            }
        }
        with (
            patch(
                "policy.SVLR.deploy_policy._world_xyz_from_camera_object",
                return_value=None,
            ),
            patch(
                "policy.SVLR.deploy_policy._render_vlm_bgr_with_shader"
            ) as render_vlm,
        ):
            payload = extract_camera_payload(
                observation,
                "right_camera",
                env=sentinel.env,
                vlm_camera_shader_dir="",
            )

        render_vlm.assert_not_called()
        self.assertNotIn("vlm_color_bgr_jpeg_b64", payload)
        self.assertIsNotNone(payload["depth_npy_b64"])

    def test_wall_transition_is_idempotent_and_completes_observation_phase(self):
        task = observe_and_pickup.__new__(observe_and_pickup)
        task.scene = sentinel.scene
        task.wall = None
        task.get_obs_cnt = 0
        task._update_render = Mock()

        with patch(
            "envs.observe_and_pickup.create_box",
            return_value=sentinel.wall,
        ) as create_box:
            first = task.on_initial_observation_published()
            second = task.on_initial_observation_published()

        self.assertIs(first, sentinel.wall)
        self.assertIs(second, sentinel.wall)
        self.assertEqual(task.get_obs_cnt, 20)
        create_box.assert_called_once()
        np.testing.assert_allclose(
            create_box.call_args.args[1].p,
            [0.0, 0.05, 1.05],
        )
        self.assertEqual(
            create_box.call_args.kwargs["half_size"],
            [0.4, 0.005, 0.3],
        )
        task._update_render.assert_called_with()


if __name__ == "__main__":
    unittest.main()
