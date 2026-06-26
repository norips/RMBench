"""
Simulation web server — RoboTwin (SAPIEN) behind the robot_server.py HTTP API
-----------------------------------------------------------------------------
Exposes the same endpoints robot_server.py exposes, so SVLR's internal robot
client talks to the RoboTwin simulation with no change. Adds an optional driver
thread that calls SVLR's Gradio API (/process_llm_command) once per episode.

Topology
--------
    RoboTwin harness (this module is its "policy": get_model/eval/reset_model)
        │  serves cam/pose/actions on :65500  ◀── SVLR's robot client
    SVLR Gradio app (:7860)  ── the brain ──────┘
        ▲
    driver thread (in this process) ── /process_llm_command(instruction) per episode

Why there is no startup deadlock
--------------------------------
"Port listening" != "server has data". SimServer.start() binds :65500 during
get_model(), before the episode loop and independent of SVLR. The bridge is
pre-seeded with a placeholder frame + neutral pose, so /camera and /robot_pose
answer from the instant the port binds — SVLR can boot in any order. The driver
retries connecting to :7860, so order is fully forgiving.

Per-step policy contract (from eval_policy)
-------------------------------------------
    reset_func(model)
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        observation = TASK_ENV.get_obs()
        eval_func(TASK_ENV, model, observation)   # ONE call == ONE step
        if TASK_ENV.eval_success: break

One eval() step:
    publish harness frame + measured pose  ->  set end_action True (sim ready)
    ->  block until SVLR POSTs /send_action  ->  drive TASK_ENV.take_action(...,"ee")
    repeatedly until the EE POSITION reaches the requested XYZ, then return.
Because send_action sets end_action False and it only flips True again at the NEXT
step's publish (after step() returns), SVLR cannot observe completion until the
end-effector is roughly in place. Publishing before flipping end_action also keeps
SVLR from ever reading a stale frame.

Threading: eval()/reset_model() run on the harness MAIN thread (all env access
stays there); uvicorn + the SVLR driver run on background daemon threads; HTTP
endpoints only read cached bridge state.

Install: put at policy/<policy_name>.py and set policy_name in the eval config.
Config via usr_args (sim_host/sim_port/sim_arm/sim_drive/svlr_url) or env
(SIM_HOST/SIM_PORT/SIM_ARM/SIM_DRIVE/SVLR_URL).
Smoke test without SAPIEN:  python sim_server.py --mock  [--drive]

ADAPT to your build: extract_camera_payload, _endpose_from_obs (+layout/quat),
sim_list_entities.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
import queue
import threading
import time
from io import BytesIO
from typing import Any, Optional

import cv2 as cv
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image


# ===========================================================================
# Pydantic models (same shape SVLR already consumes)
# ===========================================================================


class ActionPayload(BaseModel):
    model_config = {"extra": "allow"}


class SegmentEntityPayload(BaseModel):
    name: str


class OkResponse(BaseModel):
    ok: bool = True


class EndActionResponse(BaseModel):
    end_action: bool


class PoseResponse(BaseModel):
    pose: list[float] | None


class StatusResponse(BaseModel):
    action_count: int
    end_action: bool
    pose: list[float] | None
    instruction: str | None
    done: bool
    success: bool
    mode: str


# ===========================================================================
# ====================  ADAPT THESE TO YOUR ROBOTWIN BUILD  =================
# ===========================================================================
#
# Endpose ('ee') layout: [ left_xyz(3) left_quat(4) left_grip(1)
#                          right_xyz(3) right_quat(4) right_grip(1) ] -> 16
ENDPOSE_DIM = 16
_ARM_BASE = {"left": 0, "right": 8}
CONTROLLED_ARM = "right"            # which arm SVLR's single EE target drives
FIXED_QUAT_WXYZ = (0.7035625528174423, -6.977925221139638e-06, -3.883136134669406e-06, 0.7106333331678397)
QUAT_ORDER = "wxyz"                 # quat order in each *_endpose entry + take_action; SAPIEN=wxyz
CAMERA_KEY = "right_camera"
PLACEHOLDER_W, PLACEHOLDER_H = 640, 480

DEFAULT_ENDPOSE = np.array(
    [-2.44753662e-06, -2.09625375e-01,  1.23524601e+00,  5.31254846e-01, -4.66658013e-01,  4.66638342e-01,  5.31269465e-01, 1.00000000e+00,
     -2.44753662e-06, -2.09625375e-01,  1.23524601e+00,  5.31254846e-01, -4.66658013e-01,  4.66638342e-01,  5.31269465e-01, 1.00000000e+00], dtype=np.float64,
)

# --- Completion gating: end_action only after the EE POSITION reaches its target ---
# A single take_action may not have driven the end-effector all the way to the
# commanded XYZ yet, so step() keeps stepping (re-issuing the same target) until
# the measured EE position is within EE_POS_TOL_M of the requested one. This
# mirrors RealRobotBackend's completion check (COMPLETION_THRESHOLD_M / HOLD).
EE_POS_TOL_M = 0.02          # measured-to-target EE distance (m) treated as "reached" (~20 mm)
EE_SETTLE_EPS_M = 0.002      # per-substep EE motion (m) below this -> "settled" (can't get closer)
EE_HOLD_FRAMES = 3           # consecutive passing checks required before completing
MAX_SUBSTEPS_PER_ACTION = 60  # hard cap on take_action calls per SVLR command (anti-deadlock)

# --- Fixed "home"/ready pose driven on the first step of each episode, BEFORE
# SVLR is invoked (the sim analogue of RealRobotBackend._move_to_initial_position).
# Controlled-arm target: xyz(3) + quat(4, in QUAT_ORDER) + gripper(1).  ADAPT.
HOME_ON_RESET = True
HOME_CONTROLLED = np.array([0, -0.2,  1.25,  0.5, -0.5, 0.5, 0.5, 1.0], dtype=np.float64)


def map_gripper(svlr_gripper: float) -> float:
    """SVLR's gripper scalar -> RoboTwin endpose gripper convention. ADAPT.

    Identity by default. If SVLR sends SO-100 units (e.g. ~1.6 rad) but RoboTwin's
    endpose gripper channel is normalized [0,1] (or a width), convert here. This
    affects the gripper COMMAND only; completion is gated on EE position, not the
    gripper, so unit mismatches here won't stall the handshake.
    """
    return float(svlr_gripper)


def _quat_wxyz_to_xyzw(q) -> list[float]:
    if QUAT_ORDER == "xyzw":
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _jpeg_b64(bgr: np.ndarray) -> str:
    ok, jpg = cv.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def _intrinsics(w: int, h: int, fx=1.0, fy=1.0, ppx=None, ppy=None) -> dict:
    return {"width": int(w), "height": int(h), "fx": fx, "fy": fy,
            "ppx": w / 2.0 if ppx is None else ppx,
            "ppy": h / 2.0 if ppy is None else ppy,
            "model": "none", "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]}


def placeholder_camera_payload(w: int = PLACEHOLDER_W, h: int = PLACEHOLDER_H) -> dict:
    """A valid (black) frame so /camera/rgbd never 503s before episode 0.
    Removes the boot-time coupling between SVLR and the sim server."""
    black = np.zeros((h, w, 3), np.uint8)
    return {"ok": True, "color_bgr_jpeg_b64": _jpeg_b64(black), "depth_npy_b64": None,
            "intrinsics": _intrinsics(w, h), "camera_name": "placeholder",
            "timestamp_s": time.time(), "placeholder": True}


def extract_camera_payload(obs: Any, camera_key: str = CAMERA_KEY) -> dict[str, Any]:
    """obs -> LocalCameraSource.read_payload shape. ADAPT navigation/keys.
    Runs on the harness thread on a materialized obs (no env/SAPIEN call)."""
    cam = obs["observation"][camera_key]
    rgb = np.asarray(cam["rgb"])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    depth_b64 = None
    depth = cam.get("depth") if isinstance(cam, dict) else None
    if depth is not None:
        buf = BytesIO()
        np.save(buf, np.asarray(depth, dtype=np.float32))
        depth_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    h, w = rgb.shape[:2]
    intr = cam.get("intrinsic_cv") if isinstance(cam, dict) else None
    if intr is not None:
        intr = np.asarray(intr, dtype=np.float64)
        intrinsics = _intrinsics(w, h, float(intr[0, 0]), float(intr[1, 1]),
                                 float(intr[0, 2]), float(intr[1, 2]))
    else:
        intrinsics = _intrinsics(w, h)

    return {"ok": True, "color_bgr_jpeg_b64": _jpeg_b64(bgr), "depth_npy_b64": depth_b64,
            "intrinsics": intrinsics, "camera_name": str(camera_key),
            "timestamp_s": time.time()}


def _entry_to_xyz_quat(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """One endpose entry (left_endpose / right_endpose) -> (xyz[3], quat[4]) with
    the quat in QUAT_ORDER. The EE-position gate uses xyz only, so quat handling
    is best-effort. ADAPT if your pose is not xyz+quaternion.

    Handles: length>=7 (xyz + quat, default), length 6 (xyz + euler xyz),
    length 3 (xyz only), 4x4 homogeneous matrix.
    """
    a = np.asarray(pose, dtype=np.float64).reshape(-1)
    if a.size == 16:  # 4x4 matrix
        T = a.reshape(4, 4)
        xyz = T[:3, 3].copy()
        try:
            from scipy.spatial.transform import Rotation as R
            q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
            quat = q_xyzw if QUAT_ORDER == "xyzw" else np.array([q_xyzw[3], *q_xyzw[:3]])
        except Exception:
            quat = np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
        return xyz, quat
    xyz = a[:3].copy()
    if a.size >= 7:
        quat = a[3:7].copy()                 # assume already in QUAT_ORDER
    elif a.size == 6:
        try:
            from scipy.spatial.transform import Rotation as R
            q_xyzw = R.from_euler("xyz", a[3:6]).as_quat()
            quat = q_xyzw if QUAT_ORDER == "xyzw" else np.array([q_xyzw[3], *q_xyzw[:3]])
        except Exception:
            quat = np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
    else:
        quat = np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
    return xyz, quat


def _endpose_from_obs(obs: Any) -> Optional[np.ndarray]:
    """obs -> internal 16-dim endpose [Lxyz Lquat Lgrip Rxyz Rquat Rgrip], or None.

    Your obs["endpose"] is a dict: left_endpose / left_gripper / right_endpose /
    right_gripper. Also tolerates a flat 16/14 array for other configs.
    """
    try:
        ep = obs["endpose"]
    except (KeyError, TypeError):
        return None

    out = np.empty(ENDPOSE_DIM, dtype=np.float64)
    if isinstance(ep, dict):
        try:
            lx, lq = _entry_to_xyz_quat(ep["left_endpose"])
            rx, rq = _entry_to_xyz_quat(ep["right_endpose"])
            lg = float(np.asarray(ep["left_gripper"], dtype=np.float64).reshape(-1)[0])
            rg = float(np.asarray(ep["right_gripper"], dtype=np.float64).reshape(-1)[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        out[0:3], out[3:7], out[7] = lx, lq, lg
        out[8:11], out[11:15], out[15] = rx, rq, rg
        return out

    # Fallback: flat array layouts.
    try:
        flat = np.asarray(ep, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.size == ENDPOSE_DIM:
        return flat.copy()
    if flat.size == 14:
        out[:] = DEFAULT_ENDPOSE
        out[0:7], out[8:15] = flat[0:7], flat[7:14]
        return out
    return None


def build_take_action(svlr_action: dict, cmd: np.ndarray, arm: str = CONTROLLED_ARM) -> np.ndarray:
    base = _ARM_BASE[arm]
    out = cmd.copy()
    pos = svlr_action.get("pos_end_effector") or [svlr_action["ee.x"], svlr_action["ee.y"], svlr_action["ee.z"]]
    out[base + 0:base + 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
    out[base + 3:base + 7] = [float(pos[3]), float(pos[4]), float(pos[5]), float(pos[6])] if len(pos) >= 7 else np.array(FIXED_QUAT_WXYZ, dtype=np.float64)
    if "gripper" in svlr_action or "ee.gripper_pos" in svlr_action:
        raw = svlr_action.get("gripper", svlr_action.get("ee.gripper_pos"))
        out[base + 7] = map_gripper(float(raw))  # SVLR units -> RoboTwin gripper units
    # else: keep the previous gripper value already in `out`
    return out


def requested_ee_xyz(cmd: np.ndarray, arm: str = CONTROLLED_ARM) -> np.ndarray:
    base = _ARM_BASE[arm]
    return cmd[base:base + 3].astype(np.float64).copy()


def measured_ee_xyz(obs: Any, arm: str = CONTROLLED_ARM) -> Optional[np.ndarray]:
    """Measured EE position of the controlled arm from obs, or None if unavailable."""
    m = _endpose_from_obs(obs)
    if m is None:
        return None
    base = _ARM_BASE[arm]
    return m[base:base + 3].astype(np.float64).copy()


def pose_for_svlr(cmd: np.ndarray, arm: str = CONTROLLED_ARM) -> list[float]:
    base = _ARM_BASE[arm]
    xyz, quat = cmd[base:base + 3], cmd[base + 3:base + 7]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *_quat_wxyz_to_xyzw(quat)]


def sim_list_entities(env: Any) -> set[str]:
    names: set[str] = set()
    try:
        for actor in env.scene.get_all_actors():
            names.add(actor.get_name())
    except Exception:
        pass
    return names


# ===========================================================================
# Thread-safe bridge (pre-seeded so it answers from bind-time)
# ===========================================================================


class SimBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action_q: "queue.Queue[dict]" = queue.Queue()
        self._end_action = False
        self._instruction: Optional[str] = None
        # Pre-seed: server is fully answerable before episode 0 -> no boot deadlock.
        self._pose: list[float] = pose_for_svlr(DEFAULT_ENDPOSE)
        self._camera: dict = placeholder_camera_payload()
        self._entities: set[str] = set()
        self._action_count = 0
        self._done = False
        self._success = False
        self.stop_requested = False

    # producer (harness/main thread)
    def begin_episode(self) -> None:
        with self._lock:
            with self._action_q.mutex:
                self._action_q.queue.clear()   # drop any stale actions from the prior episode
            self._end_action = False
            self._done = self._success = False
            self._action_count = 0
            self.stop_requested = False

    def publish(self, pose, camera, entities) -> None:
        with self._lock:
            self._pose, self._camera, self._entities = pose, camera, entities

    def set_end_action(self, v: bool) -> None:
        with self._lock:
            self._end_action = v

    def set_instruction(self, s: str) -> None:
        with self._lock:
            self._instruction = s

    def set_done(self, success: bool) -> None:
        with self._lock:
            self._done, self._success = True, success

    def pop_action(self, timeout: float) -> Optional[dict]:
        try:
            return self._action_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # consumer (uvicorn thread)
    def send_action(self, payload: dict) -> None:
        with self._lock:
            self._end_action = False
            self._action_count += 1
        self._action_q.put(payload)

    def reset_end_action(self) -> None:
        with self._lock:
            self._end_action = False

    def is_end_action(self) -> bool:
        with self._lock:
            return self._end_action

    def get_pose(self):
        with self._lock:
            return list(self._pose)

    def get_camera(self):
        with self._lock:
            return self._camera

    def segment(self, name: str) -> bool:
        with self._lock:
            return name in self._entities

    def status(self) -> dict:
        with self._lock:
            return {"action_count": self._action_count, "end_action": self._end_action,
                    "pose": list(self._pose), "instruction": self._instruction,
                    "done": self._done, "success": self._success}


# ===========================================================================
# FastAPI app
# ===========================================================================


def create_app(bridge: SimBridge) -> FastAPI:
    app = FastAPI(title="Simulation web server", version="1.0.0")

    @app.get("/robot_pose", response_model=PoseResponse)
    async def robot_pose():
        return {"pose": bridge.get_pose()}

    @app.get("/end_action", response_model=EndActionResponse)
    async def end_action():
        return {"end_action": bridge.is_end_action()}

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return {**bridge.status(), "mode": "sim"}

    @app.get("/instruction")
    async def instruction():
        return {"instruction": bridge.status()["instruction"]}

    @app.get("/camera/rgbd")
    async def camera_rgbd():
        payload = bridge.get_camera()
        if payload is None:
            raise HTTPException(status_code=503, detail="no frame published yet")
        return JSONResponse(payload)

    @app.post("/send_action", response_model=OkResponse)
    async def send_action(payload: ActionPayload):
        bridge.send_action(payload.model_dump())
        return {"ok": True}

    @app.post("/reset_end_action", response_model=OkResponse)
    async def reset_end_action():
        bridge.reset_end_action()
        return {"ok": True}

    @app.post("/segment_entity")
    async def segment_entity(payload: SegmentEntityPayload):
        return {"found": bridge.segment(payload.name), "name": payload.name}

    @app.post("/stop", response_model=OkResponse)
    async def stop():
        bridge.stop_requested = True   # ends current episode; cleared on next reset
        return {"ok": True, "message": "stopping current episode"}

    return app


# ===========================================================================
# Server + per-step driver + optional SVLR Gradio driver thread
# ===========================================================================


class SimServer:
    def __init__(self, host="0.0.0.0", port=65500, controlled_arm=CONTROLLED_ARM,
                 action_poll_s=0.1, drive=False, svlr_url="http://127.0.0.1:7860",
                 ee_pos_tol_m=EE_POS_TOL_M, ee_settle_eps_m=EE_SETTLE_EPS_M,
                 ee_hold_frames=EE_HOLD_FRAMES,
                 max_substeps_per_action=MAX_SUBSTEPS_PER_ACTION,
                 home_on_reset=HOME_ON_RESET, home_controlled=None) -> None:
        self.host, self.port = host, port
        self.controlled_arm = controlled_arm
        self.action_poll_s = action_poll_s
        self.drive, self.svlr_url = drive, svlr_url
        self.ee_pos_tol_m = ee_pos_tol_m
        self.ee_settle_eps_m = ee_settle_eps_m
        self.ee_hold_frames = ee_hold_frames
        self.max_substeps_per_action = max_substeps_per_action
        self.home_on_reset = home_on_reset
        self.home_controlled = np.asarray(
            HOME_CONTROLLED if home_controlled is None else home_controlled, dtype=np.float64
        ).reshape(-1)
        assert self.home_controlled.size == 8, "home_controlled must be xyz(3)+quat(4)+gripper(1)"
        self.bridge = SimBridge()
        self.app = create_app(self.bridge)
        self._cmd: Optional[np.ndarray] = None
        self._uv: Optional[uvicorn.Server] = None
        self._uv_thread: Optional[threading.Thread] = None
        self._episode_q: "queue.Queue[str]" = queue.Queue()
        self._driver_thread: Optional[threading.Thread] = None
        self._shutdown = False

    def start(self) -> None:
        if self._uv_thread is not None:
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._uv = uvicorn.Server(config)
        self._uv.install_signal_handlers = lambda: None
        self._uv_thread = threading.Thread(target=self._uv.run, daemon=True)
        self._uv_thread.start()
        while not getattr(self._uv, "started", False):
            time.sleep(0.01)
        print(f"[sim-server] http://{self.host}:{self.port}  (arm: {self.controlled_arm})  "
              f"drive={'on->' + self.svlr_url if self.drive else 'off'}")
        if self.drive:
            self._driver_thread = threading.Thread(target=self._drive_loop, daemon=True)
            self._driver_thread.start()

    # -- SVLR driver: retries connect, fires one /process_llm_command per episode --
    def _drive_loop(self) -> None:
        try:
            from gradio_client import Client
        except ImportError:
            print("[driver] gradio_client not installed; SVLR auto-drive disabled")
            return
        client = None
        while not self._shutdown:
            if client is None:
                try:
                    client = Client(self.svlr_url)
                    print(f"[driver] connected to SVLR at {self.svlr_url}")
                except Exception:
                    time.sleep(1.0)
                    continue
            try:
                instruction = self._episode_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # Blocks for the whole episode while SVLR drives :65500.
                client.predict(prompt=instruction, api_name="/process_llm_command")
            except Exception as e:
                print(f"[driver] SVLR command failed: {e}")
                client = None  # force reconnect next round

    # -- per-episode reset --
    def reset_episode(self) -> None:
        self._cmd = None
        self.bridge.begin_episode()

    # -- one eval() == one step --
    def step(self, env: Any, observation: Any) -> None:
        cam = env.get_obs()["observation"][CAMERA_KEY]
        rgb = np.asarray(cam["rgb"])
        img = Image.fromarray(rgb)
        img.save(f"output_step{env.take_action_cnt}.png")
        cam = env.get_obs()["observation"]["right_camera"]
        rgb = np.asarray(cam["rgb"])
        img = Image.fromarray(rgb)
        img.save(f"output_right_step{env.take_action_cnt}.png")
        print(f"[sim-server] save output_step{env.take_action_cnt}.png")
        
        print("[sim-server] end pose:", _endpose_from_obs(observation))
        
        
        
        if self._cmd is None:
            # First step of the episode: home the arm BEFORE engaging SVLR, then
            # publish the homed frame and hand the instruction to the driver.
            self._home_and_engage(env, observation)
            cam = env.get_obs()["observation"][CAMERA_KEY]
            rgb = np.asarray(cam["rgb"])
            img = Image.fromarray(rgb)
            img.save(f"output_step_home{env.take_action_cnt}.png")
            cam = env.get_obs()["observation"]["right_camera"]
            rgb = np.asarray(cam["rgb"])
            img = Image.fromarray(rgb)
            img.save(f"output_right_step_home{env.take_action_cnt}.png")
            print(f"[sim-server] save output_step_home{env.take_action_cnt}.png")
        else:
            measured = _endpose_from_obs(observation)
            pose = pose_for_svlr(measured if measured is not None else self._cmd, self.controlled_arm)
            self.bridge.publish(pose, extract_camera_payload(observation), sim_list_entities(env))

        action = None
        while not self.bridge.stop_requested:
            action = self.bridge.pop_action(timeout=self.action_poll_s)
            if action is not None:
                break
        if action is None:  # /stop -> end episode by forcing the harness loop's exit
            with contextlib.suppress(Exception):
                env.take_action_cnt = env.step_lim
            return

        self._cmd = build_take_action(action, self._cmd, self.controlled_arm)
        run_for = self._execute_until_ee_reached(env)
        print(f"[sim-server] take_action: {self._cmd}  (ran {run_for} substeps)")
        self.bridge.set_end_action(True)

        if bool(getattr(env, "eval_success", False)):
            self.bridge.set_done(True)
            with contextlib.suppress(Exception):
                self.bridge.publish(pose_for_svlr(self._cmd, self.controlled_arm),
                                    extract_camera_payload(env.get_obs()), sim_list_entities(env))
                self.bridge.set_end_action(True)

    # -- first step: drive to the fixed home pose, then publish + engage SVLR --
    def _home_and_engage(self, env: Any, observation: Any) -> None:
        # Seed the command from the measured pose so the un-driven arm is held put.
        measured = _endpose_from_obs(observation)
        self._cmd = (measured if measured is not None else DEFAULT_ENDPOSE).copy()

        if self.home_on_reset:
            print(f"[sim-server] homing {self.controlled_arm} arm to {self.home_controlled}")
            base = _ARM_BASE[self.controlled_arm]
            self._cmd[base:base + 8] = self.home_controlled   # xyz + quat + gripper
            self._execute_until_ee_reached(env)               # move to home (consumes a few steps)

        # Publish a FRESH post-home frame (the passed `observation` is now stale),
        # flag ready, THEN release SVLR so it acts on the homed state.
        pub_obs = env.get_obs()
        m = _endpose_from_obs(pub_obs)
        pose = pose_for_svlr(m if m is not None else self._cmd, self.controlled_arm)
        self.bridge.publish(pose, extract_camera_payload(pub_obs), sim_list_entities(env))
        self.bridge.set_end_action(True)

        with contextlib.suppress(Exception):
            instr = env.get_instruction()
            self.bridge.set_instruction(instr)
            if self.drive:
                self._episode_q.put(instr)   # driver fires /process_llm_command now

    # -- step the sim until the EE position has reached the requested target --
    def _execute_until_ee_reached(self, env: Any) -> None:
        """Re-issue the current target until the measured EE position is within
        ee_pos_tol_m of the requested XYZ (or the arm stops moving, or caps are
        hit). Only after this returns will the next step flip end_action True, so
        SVLR never sees completion before the end-effector is roughly in place.

        Completion = EE within tol of target, OR EE stopped moving (can't get
        closer — IK/contact limit), held for ee_hold_frames checks. Falls back to
        max_substeps_per_action / step_lim to guarantee termination.
        """
        target = requested_ee_xyz(self._cmd, self.controlled_arm)
        step_lim = getattr(env, "step_lim", None)
        prev = None
        hold = 0

        for iter in range(self.max_substeps_per_action):
            # print(f"[sim-server] take_action: {self._cmd}")
            env.take_action(self._cmd, action_type="ee")
            
            observation = env.get_obs()
            measured = _endpose_from_obs(observation)
            pose = pose_for_svlr(measured if measured is not None else self._cmd, self.controlled_arm)
            self.bridge.publish(pose, extract_camera_payload(observation), sim_list_entities(env))

            meas = measured_ee_xyz(env.get_obs(), self.controlled_arm)
            # print("[sim-server] measured EE xyz:", meas, "target:", target)
            if meas is None:
                return iter  # EE not observable -> cannot gate on it; complete now

            dist = float(np.linalg.norm(meas - target))
            moved = None if prev is None else float(np.linalg.norm(meas - prev))
            prev = meas

            reached = dist <= self.ee_pos_tol_m
            settled = moved is not None and moved <= self.ee_settle_eps_m

            if reached or settled:
                hold += 1
                if hold >= self.ee_hold_frames:
                    return iter
            else:
                hold = 0

            if bool(getattr(env, "eval_success", False)):
                return iter
            if step_lim is not None and getattr(env, "take_action_cnt", 0) >= step_lim:
                return iter


# ===========================================================================
# RoboTwin eval-harness entry points
# ===========================================================================

_SERVER: Optional[SimServer] = None


def _cfg(usr_args, key, env_key, default, cast):
    if usr_args and key in usr_args and usr_args[key] is not None:
        return cast(usr_args[key])
    v = os.environ.get(env_key)
    return cast(v) if v is not None else default


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_vec(v):
    """Accept a list/tuple or a comma-separated string -> list[float]."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(x) for x in str(v).split(",")]


def get_model(usr_args=None):
    global _SERVER
    if _SERVER is None:
        home_controlled = None
        if usr_args and usr_args.get("sim_home_controlled") is not None:
            home_controlled = _as_vec(usr_args["sim_home_controlled"])
        elif os.environ.get("SIM_HOME_CONTROLLED"):
            home_controlled = _as_vec(os.environ["SIM_HOME_CONTROLLED"])
        _SERVER = SimServer(
            host=_cfg(usr_args, "sim_host", "SIM_HOST", "0.0.0.0", str),
            port=_cfg(usr_args, "sim_port", "SIM_PORT", 65500, int),
            controlled_arm=_cfg(usr_args, "sim_arm", "SIM_ARM", CONTROLLED_ARM, str),
            drive=_cfg(usr_args, "sim_drive", "SIM_DRIVE", True, _as_bool),
            svlr_url=_cfg(usr_args, "svlr_url", "SVLR_URL", "http://127.0.0.1:7860", str),
            ee_pos_tol_m=_cfg(usr_args, "sim_ee_pos_tol", "SIM_EE_POS_TOL", EE_POS_TOL_M, float),
            max_substeps_per_action=_cfg(usr_args, "sim_max_substeps", "SIM_MAX_SUBSTEPS",
                                         MAX_SUBSTEPS_PER_ACTION, int),
            home_on_reset=_cfg(usr_args, "sim_home", "SIM_HOME", HOME_ON_RESET, _as_bool),
            home_controlled=home_controlled,
        )
        _SERVER.start()
    return _SERVER


def reset_model(model):
    model.reset_episode()


def encode_obs(observation):
    return observation


def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, encode_obs(observation))


# ===========================================================================
# Mock env + harness loop (smoke test, no SAPIEN)
# ===========================================================================


class MockSimEnv:
    EE_STEP_M = 0.05  # EE moves up to this far per take_action (mimics a finite arm speed)

    def __init__(self, w=640, h=480, step_lim=100000):
        self.w, self.h, self.step_lim = w, h, step_lim
        self.take_action_cnt = 0
        self.eval_success = False
        self._endpose = DEFAULT_ENDPOSE.copy()

    def get_instruction(self):
        return "pick up the block and place it on the plate"

    def take_action(self, action, action_type="ee"):
        target = np.asarray(action, dtype=np.float64).reshape(-1)
        # Move each arm's EE xyz toward the target by at most EE_STEP_M; snap the
        # orientation + gripper. This makes the EE-position gate take a few substeps.
        for base in (0, 8):
            cur = self._endpose[base:base + 3]
            d = target[base:base + 3] - cur
            n = float(np.linalg.norm(d))
            cur += d if n <= self.EE_STEP_M else d * (self.EE_STEP_M / n)
            self._endpose[base + 3:base + 8] = target[base + 3:base + 8]
        self.take_action_cnt += 1
        rx, ry, rz = self._endpose[8:11]
        print(f"[mock-sim] take_action ({action_type}) cnt={self.take_action_cnt} "
              f"ee_r=({rx:.3f},{ry:.3f},{rz:.3f})")

    def get_obs(self):
        t = self.take_action_cnt
        x = np.linspace(0, 255, self.w, dtype=np.uint8)
        row = (x + t * 3) % 256
        rgb = np.stack([np.tile(row, (self.h, 1)), np.tile(row[::-1], (self.h, 1)),
                        np.full((self.h, self.w), (t * 5) % 256, np.uint8)], axis=-1).astype(np.uint8)
        return {"observation": {CAMERA_KEY: {"rgb": rgb, "depth": np.full((self.h, self.w), 0.8, np.float32)}},
                "endpose": {
                    "left_endpose": np.concatenate([self._endpose[0:3], self._endpose[3:7]]),
                    "left_gripper": float(self._endpose[7]),
                    "right_endpose": np.concatenate([self._endpose[8:11], self._endpose[11:15]]),
                    "right_gripper": float(self._endpose[15]),
                }}


def _run_mock_harness(server: SimServer):
    env = MockSimEnv()
    reset_model(server)
    print("[mock-sim] serving; drive from SVLR (--drive) or POST /stop / Ctrl-C to end.")
    while env.take_action_cnt < env.step_lim and not server.bridge.stop_requested:
        eval(env, server, env.get_obs())
        if env.eval_success:
            break
    print("[mock-sim] episode ended.")


def parse_args():
    p = argparse.ArgumentParser(description="RoboTwin simulation web server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=65500)
    p.add_argument("--arm", choices=["left", "right"], default=CONTROLLED_ARM)
    p.add_argument("--mock", action="store_true", help="run the mock env (no SAPIEN)")
    p.add_argument("--drive", action="store_true", help="auto-drive a running SVLR Gradio app")
    p.add_argument("--svlr-url", default="http://127.0.0.1:7860")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.mock:
        raise SystemExit("Standalone ships only the mock env. Set policy_name to this "
                         "module for real RoboTwin. Re-run with --mock to smoke-test.")
    global _SERVER
    _SERVER = SimServer(host=args.host, port=args.port, controlled_arm=args.arm,
                        drive=args.drive, svlr_url=args.svlr_url)
    _SERVER.start()
    try:
        _run_mock_harness(_SERVER)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()