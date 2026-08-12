#!/usr/bin/env python3
"""Validate the Franka/Panda files required by demo_clean_franka + SVLR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REQUIRED_FRANKA_FILES = [
    "config.yml",
    "panda.urdf",
    "panda.srdf",
    "curobo_tmp.yml",
    "curobo.yml",
    "curobo_left.yml",
    "curobo_right.yml",
    "collision_franka.yml",
    "franka_description/meshes/visual/link0.glb",
    "franka_description/meshes/visual/link1.glb",
    "franka_description/meshes/visual/link2.glb",
    "franka_description/meshes/visual/link3.glb",
    "franka_description/meshes/visual/link4.glb",
    "franka_description/meshes/visual/link5.glb",
    "franka_description/meshes/visual/link6.glb",
    "franka_description/meshes/visual/link7.glb",
    "franka_description/meshes/visual/hand.glb",
    "franka_description/meshes/visual/finger.glb",
    "franka_description/meshes/visual/camera_base.glb",
    "franka_description/meshes/visual/d435.dae",
    "franka_description/meshes/collision/link0.stl",
    "franka_description/meshes/collision/link1.stl",
    "franka_description/meshes/collision/link2.stl",
    "franka_description/meshes/collision/link3.stl",
    "franka_description/meshes/collision/link4.stl",
    "franka_description/meshes/collision/link5.stl",
    "franka_description/meshes/collision/link6.stl",
    "franka_description/meshes/collision/link7.stl",
    "franka_description/meshes/collision/hand.stl",
    "franka_description/meshes/collision/finger.stl",
]

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def yaml_path_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from yaml_path_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from yaml_path_values(value)
    elif isinstance(obj, str):
        yield obj


def validate_curobo_yaml(path: Path) -> list[str]:
    missing: list[str] = []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "${ASSETS_PATH}" in path.read_text(encoding="utf-8") or "$ASSETS_PATH" in path.read_text(encoding="utf-8"):
        missing.append(f"{path}: contains unresolved ASSETS_PATH placeholder")
    for value in yaml_path_values(data):
        if not isinstance(value, str):
            continue
        if not (value.startswith("/") or value.endswith((".urdf", ".yml", ".yaml", ".srdf", ".glb", ".stl", ".dae"))):
            continue
        p = Path(value)
        if not p.is_absolute():
            p = repo_root() / p
        if not p.exists():
            missing.append(f"{path}: referenced path missing: {p}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data", action="store_true", help="Do not validate RMBench evaluation data files.")
    args = parser.parse_args()

    root = repo_root()
    franka_dir = root / "assets" / "embodiments" / "franka-panda"
    missing: list[str] = []

    for rel in REQUIRED_FRANKA_FILES:
        p = franka_dir / rel
        if not p.exists():
            missing.append(str(p))

    for yml_name in ("curobo.yml", "curobo_left.yml", "curobo_right.yml"):
        yml_path = franka_dir / yml_name
        if yml_path.exists():
            missing.extend(validate_curobo_yaml(yml_path))

    config_path = franka_dir / "config.yml"
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if cfg.get("dual_arm") is not False:
            missing.append(f"{config_path}: expected dual_arm: False for one physical Franka")
        for key in ("move_group", "ee_joints", "arm_joints_name", "gripper_name", "homestate", "robot_pose"):
            value = cfg.get(key)
            if not isinstance(value, list) or len(value) < 2:
                missing.append(f"{config_path}: expected {key} to contain left/right logical entries")
        if cfg.get("robot_pose") and len(cfg["robot_pose"]) >= 2 and cfg["robot_pose"][0] != cfg["robot_pose"][1]:
            missing.append(
                f"{config_path}: franka-panda is one physical robot; robot_pose left/right logical entries must match"
            )

    if missing:
        print("[validate-franka] validation failed:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(2)

    print(f"[validate-franka] OK: {franka_dir}")
    print("[validate-franka] OK: logical left/right slots map to one physical franka-panda embodiment")


if __name__ == "__main__":
    main()
