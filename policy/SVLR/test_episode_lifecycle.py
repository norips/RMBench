import asyncio
import threading
import unittest
from unittest.mock import sentinel

from policy.SVLR.deploy_policy import (
    ActionPayload,
    SimBridge,
    SimServer,
    StatusResponse,
    create_app,
)


class SimBridgeEpisodeTests(unittest.TestCase):
    def test_episode_ids_increment_and_status_is_complete(self):
        bridge = SimBridge()

        first = bridge.begin_episode()
        second = bridge.begin_episode()
        status = bridge.status()

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        self.assertEqual(status["episode_id"], 2)
        self.assertFalse(status["accept_actions"])
        self.assertFalse(status["stop_requested"])
        self.assertEqual(status["rejected_action_count"], 0)

        public_status = StatusResponse(**status, mode="sim")
        serialized = public_status.model_dump()
        self.assertEqual(serialized["episode_id"], 2)
        self.assertIn("accept_actions", serialized)
        self.assertIn("rejected_action_count", serialized)
        self.assertIn("stop_requested", serialized)
        self.assertEqual(serialized["observation_request_id"], 0)
        self.assertEqual(serialized["observation_ready_id"], 0)

    def test_stale_action_is_rejected_without_entering_queue(self):
        bridge = SimBridge()
        first = bridge.begin_episode()
        self.assertTrue(bridge.open_action_window(first))
        self.assertTrue(
            bridge.send_action({"episode_id": first, "gripper": 1.0})
        )
        self.assertEqual(bridge.pop_action(timeout=0.01), {"gripper": 1.0})

        second = bridge.begin_episode()
        self.assertTrue(bridge.open_action_window(second))
        self.assertFalse(
            bridge.send_action({"episode_id": first, "gripper": 0.0})
        )
        self.assertIsNone(bridge.pop_action(timeout=0.01))
        self.assertEqual(bridge.status()["rejected_action_count"], 1)

    def test_legacy_action_without_episode_id_remains_compatible(self):
        bridge = SimBridge()
        episode_id = bridge.begin_episode()
        self.assertTrue(bridge.open_action_window(episode_id))

        self.assertTrue(bridge.send_action({"gripper": 1.0}))
        self.assertEqual(bridge.pop_action(timeout=0.01), {"gripper": 1.0})

    def test_observation_pose_request_is_not_counted_as_task_action(self):
        bridge = SimBridge()
        episode_id = bridge.begin_episode()
        self.assertTrue(bridge.open_action_window(episode_id))

        requested = bridge.request_observation_pose(episode_id)

        self.assertEqual(requested, {"ok": True, "request_id": 1})
        self.assertEqual(bridge.action_count(), 0)
        self.assertEqual(
            bridge.pop_action(timeout=0.01),
            {
                "_bridge_command": "prepare_observation",
                "episode_id": episode_id,
                "request_id": 1,
            },
        )
        bridge.finish_observation_pose(episode_id, 1)
        status = bridge.status()
        self.assertEqual(status["observation_request_id"], 1)
        self.assertEqual(status["observation_ready_id"], 1)
        self.assertIsNone(status["observation_error"])

    def test_stale_driver_cannot_open_action_window(self):
        bridge = SimBridge()
        first = bridge.begin_episode()
        second = bridge.begin_episode()

        self.assertFalse(bridge.open_action_window(first))
        self.assertFalse(bridge.status()["accept_actions"])
        self.assertTrue(bridge.open_action_window(second))

    def test_send_action_endpoint_reports_rejection(self):
        bridge = SimBridge()
        episode_id = bridge.begin_episode()
        app = create_app(bridge)
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/send_action"
        )

        result = asyncio.run(
            endpoint(ActionPayload(episode_id=episode_id, gripper=1.0))
        )

        self.assertEqual(result, {"ok": False})
        self.assertEqual(bridge.status()["rejected_action_count"], 1)

    def test_stop_request_is_latched_under_bridge_lock(self):
        bridge = SimBridge()
        episode_id = bridge.begin_episode()
        self.assertTrue(bridge.open_action_window(episode_id))
        self.assertTrue(
            bridge.send_action({"episode_id": episode_id, "gripper": 1.0})
        )

        bridge.request_stop()

        status = bridge.status()
        self.assertTrue(status["stop_requested"])
        self.assertFalse(status["accept_actions"])
        self.assertIsNone(bridge.pop_action(timeout=0.01))
        self.assertFalse(
            bridge.send_action({"episode_id": episode_id, "gripper": 0.0})
        )


class SimServerHandoffTests(unittest.TestCase):
    @staticmethod
    def _server():
        server = SimServer.__new__(SimServer)
        server.bridge = SimBridge()
        server.bridge.begin_episode()
        server._cmd = sentinel.command
        server.episode_handoff_timeout_s = 0.5
        server._driver_state_lock = threading.Lock()
        server._driver_finished_event = threading.Event()
        server._driver_episode_id = 0
        server._driver_started = False
        server._driver_finished = False
        server._driver_failed = False
        return server

    def test_reset_waits_for_previous_driver_then_starts_new_episode(self):
        server = self._server()
        previous_episode = server.bridge.episode_id()
        server._mark_driver_started(previous_episode)
        timer = threading.Timer(
            0.01,
            lambda: server._mark_driver_finished(previous_episode, False),
        )
        timer.start()
        try:
            new_episode = server.reset_episode(timeout_s=0.5)
        finally:
            timer.join(timeout=1.0)

        self.assertEqual(new_episode, previous_episode + 1)
        self.assertEqual(server.bridge.episode_id(), new_episode)
        self.assertIsNone(server._cmd)
        self.assertFalse(server._driver_started)
        self.assertFalse(server._driver_finished)
        self.assertFalse(server._driver_finished_event.is_set())

    def test_handoff_latches_natural_failure_before_waiting(self):
        server = self._server()
        episode_id = server.bridge.episode_id()
        server._mark_driver_started(episode_id)
        timer = threading.Timer(
            0.01,
            lambda: server._mark_driver_finished(episode_id, False),
        )
        timer.start()
        try:
            server._wait_for_driver_handoff(timeout_s=0.5)
        finally:
            timer.join(timeout=1.0)

        status = server.bridge.status()
        self.assertTrue(status["done"])
        self.assertFalse(status["success"])
        self.assertFalse(status["accept_actions"])

    def test_handoff_timeout_is_explicit(self):
        server = self._server()
        episode_id = server.bridge.episode_id()
        server._mark_driver_started(episode_id)

        with self.assertRaisesRegex(
            RuntimeError,
            rf"episode_id={episode_id} within 0.0s",
        ):
            server._wait_for_driver_handoff(timeout_s=0.001)

    def test_stale_driver_completion_is_ignored(self):
        server = self._server()
        active_episode = server.bridge.episode_id()
        server._mark_driver_started(active_episode)

        server._mark_driver_finished(active_episode - 1, True)

        self.assertEqual(server._driver_done(active_episode), (False, False))
        self.assertFalse(server._driver_finished_event.is_set())


if __name__ == "__main__":
    unittest.main()
