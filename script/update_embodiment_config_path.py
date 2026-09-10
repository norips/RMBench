#!/usr/bin/env python3
"""Generate embodiment Curobo YAML files from *_tmp.yml templates.

The generated files contain absolute paths for the current checkout.  The script
is non-interactive and idempotent, so it is safe to run from `pixi run setup` and
safe to re-run after moving or recloning the repository.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

BLUE = "\033[0;34m"
YELLOW = "\033[0;33m"
GREEN = "\033[0;32m"
RESET = "\033[0m"

DEFAULT_VALIDATE = ["aloha-agilex", "franka-panda"]
REQUIRED_BY_EMBODIMENT = {
    "aloha-agilex": [
        "config.yml",
        "urdf/arx5_description_isaac.urdf",
        "srdf/arx5_description_isaac.srdf",
        "curobo_left_tmp.yml",
        "curobo_right_tmp.yml",
        "curobo_left.yml",
        "curobo_right.yml",
        "collision_aloha_left.yml",
        "collision_aloha_right.yml",
        "meshes/base_link.STL",
        "meshes/d435.dae",
    ],
    "franka-panda": [
        "config.yml",
        "panda.urdf",
        "panda.srdf",
        "curobo_tmp.yml",
        "curobo.yml",
        "curobo_left.yml",
        "curobo_right.yml",
        "collision_franka.yml",
        "franka_description/meshes/visual/link0.glb",
        "franka_description/meshes/visual/link7.glb",
        "franka_description/meshes/visual/camera_base.glb",
        "franka_description/meshes/visual/d435.dae",
        "franka_description/meshes/collision/link0.stl",
        "franka_description/meshes/collision/link7.stl",
        "franka_description/meshes/collision/hand.stl",
        "franka_description/meshes/collision/finger.stl",
    ],
}


# Embodiment base placement, applied to config.yml on every configure run.
#
# config.yml lives under assets/, which is downloaded and gitignored, so a pose
# edited by hand there is lost on the next fresh setup and is recorded nowhere.
# SVLR's aloha calibration -- workspace_bounds_m, command_z_bounds_m and the
# reachability reasoning in RMBENCH_ALOHA_action.json -- is all measured against
# this placement, so it has to come from a versioned file.
ROBOT_POSE_BY_EMBODIMENT = {
    "aloha-agilex": (
        "# Base translated -0.306 m in world X so the RIGHT arm is centred on the table\n"
        "# (1.2 x 0.7, centred at the origin) instead of sitting off its +X end: the\n"
        "# right shoulder moves from x=+0.306 to x=0.000. With the fingers pointing\n"
        "# down the arm reaches 0.59 m in any direction from that shoulder, which the\n"
        "# old placement left 23 cm short of the far -X corner of the workspace.\n"
        "# Y is untouched on purpose -- the shoulder is already only 0.07 m behind the\n"
        "# table's near edge, so moving forward would put the arm column over the top.\n",
        "robot_pose: [[-0.306, -0.65, 0.25, 0.707, 0, 0, 0.707]]",
    ),
}


def apply_robot_poses(repo_root: Path) -> None:
    """Force each configured embodiment's robot_pose in its config.yml.

    Only the embodiments named above are touched, and only their robot_pose
    line; everything else in the file is left exactly as downloaded.
    """
    for name, (comment, pose_line) in ROBOT_POSE_BY_EMBODIMENT.items():
        config = repo_root / "assets" / "embodiments" / name / "config.yml"
        if not config.exists():
            print_color(
                f"[embodiment-config] {name}: no config.yml yet, skipping robot_pose",
                YELLOW,
            )
            continue
        lines = config.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if not line.startswith("robot_pose:")]
        block = comment + pose_line + "\n"
        if "".join(kept) == "".join(lines) and block in "".join(lines):
            continue
        # Drop a previously written copy of the comment so re-runs do not stack it.
        text = "".join(kept).replace(comment, "")
        if len(kept) == len(lines):
            text = text.rstrip("\n") + "\n" + block
        else:
            index = next(i for i, line in enumerate(lines) if line.startswith("robot_pose:"))
            head = "".join(kept[:index]).replace(comment, "")
            tail = "".join(kept[index:])
            text = head + block + tail
        if text != "".join(lines):
            config.write_text(text, encoding="utf-8")
            print_color(f"[embodiment-config] {name}: robot_pose set", BLUE)


# Joint limits forced into an embodiment's URDF on every configure run.
#
# The shipped arx5_description_isaac.urdf gives every revolute joint a
# placeholder range of +/-10 rad (+/-573 deg). Curobo reads its limits from that
# URDF, so it is free to return solutions more than a full turn from the real
# mechanism -- measured on cover_blocks, fr_joint3 reached 4.49 rad. Those poses
# are kinematically valid to the planner and physically jammed on the robot: the
# arm then ends up to 0.48 rad from the configuration its own plan asked for and
# misses the commanded pose by ~90 mm, while Curobo still reports success.
#
# The real range comes from aloha_new.urdf in the same directory, mirrored into
# this URDF's sign convention: aloha_new declares fr_joint3 as [-2.697, 0], and
# 310 observed configurations of the running arm put it in [0, +1.418], so the
# two models run this joint in opposite directions.
#
# Constraining it removed every one of the 12 catastrophic misses on a 5-episode
# cover_blocks run (max pose error 93 mm -> 22 mm, 0/5 -> 3/5 episodes), at the
# cost of a slightly looser median (1.8 mm -> 5.9 mm) because the planner can no
# longer pick the wrapped IK branch.
#
# Like the robot_pose above, this lives here because assets/ is downloaded and
# gitignored: edited by hand it is lost on the next fresh setup.
JOINT_LIMITS_BY_EMBODIMENT = {
    "aloha-agilex": {
        "urdf/arx5_description_isaac.urdf": {
            "fr_joint3": (0.0, 2.697),
        },
    },
}


def apply_joint_limits(repo_root: Path) -> None:
    """Force the configured <limit lower/upper> for named joints.

    Only the lower and upper attributes of the named joints are touched; the
    effort and velocity on the same tag, and every other joint in the file, are
    left exactly as downloaded.
    """
    for name, files in JOINT_LIMITS_BY_EMBODIMENT.items():
        for rel_path, joints in files.items():
            urdf = repo_root / "assets" / "embodiments" / name / rel_path
            if not urdf.exists():
                print_color(
                    f"[embodiment-config] {name}: {rel_path} not present yet, "
                    "skipping joint limits",
                    YELLOW,
                )
                continue
            text = urdf.read_text(encoding="utf-8")
            original = text
            for joint, (lower, upper) in joints.items():
                pattern = re.compile(
                    r'(<joint name="' + re.escape(joint) + r'" type="revolute">.*?'
                    r'<limit lower=")[-0-9.eE]+(" upper=")[-0-9.eE]+(")',
                    re.DOTALL,
                )
                text, count = pattern.subn(
                    lambda m: f"{m.group(1)}{lower}{m.group(2)}{upper}{m.group(3)}",
                    text,
                    count=1,
                )
                if count == 0:
                    print_color(
                        f"[embodiment-config] {name}: no revolute joint "
                        f"{joint!r} with a <limit> in {rel_path}",
                        YELLOW,
                    )
            if text != original:
                urdf.write_text(text, encoding="utf-8")
                print_color(
                    f"[embodiment-config] {name}: joint limits set in {rel_path}",
                    BLUE,
                )


def print_color(message: str, color_code: str) -> None:
    print(f"{color_code}{message}{RESET}")


def repo_root_from_args(value: str | None) -> Path:
    root = Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[1]
    if not (root / "assets" / "embodiments").is_dir():
        raise SystemExit(
            f"Cannot find assets/embodiments under {root}. Run this script from an RMBench checkout "
            "or pass --repo-root."
        )
    return root


def render_template(tmp_file: Path, repo_root: Path) -> str:
    content = tmp_file.read_text(encoding="utf-8")
    repo_root_str = str(repo_root)
    return (
        content.replace("${ASSETS_PATH}", repo_root_str)
        .replace("$ASSETS_PATH", repo_root_str)
        .replace("${RMBENCH_ROOT}", repo_root_str)
        .replace("$RMBENCH_ROOT", repo_root_str)
    )


def generated_targets_for_template(tmp_file: Path) -> list[Path]:
    stem = tmp_file.name[:-8]  # strip _tmp.yml
    if stem == "curobo":
        return [tmp_file.with_name("curobo.yml"), tmp_file.with_name("curobo_left.yml"), tmp_file.with_name("curobo_right.yml")]
    return [tmp_file.with_name(f"{stem}.yml")]


def generate_configs(repo_root: Path) -> list[Path]:
    os.environ["ASSETS_PATH"] = str(repo_root)
    print_color(f"[embodiment-config] ASSETS_PATH={repo_root}", BLUE)

    templates = [
        path
        for path in sorted((repo_root / "assets" / "embodiments").glob("**/*_tmp.yml"))
        if "__MACOSX" not in path.parts and not any(part.startswith("._") for part in path.parts)
    ]
    if not templates:
        raise SystemExit(f"No *_tmp.yml files found under {repo_root / 'assets' / 'embodiments'}")

    written: list[Path] = []
    for tmp_file in templates:
        rendered = render_template(tmp_file, repo_root)
        for target_file in generated_targets_for_template(tmp_file):
            target_file.write_text(rendered, encoding="utf-8")
            written.append(target_file)
            print(f"[embodiment-config] wrote {target_file.relative_to(repo_root)}")
    print_color(f"[embodiment-config] generated {len(written)} config files", GREEN)
    return written


def _iter_path_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_path_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_path_values(value)
    elif isinstance(obj, str):
        yield obj


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("./")
        or value.startswith("../")
        or ".urdf" in value
        or ".yml" in value
        or ".yaml" in value
        or ".srdf" in value
        or ".glb" in value
        or ".stl" in value
        or ".dae" in value
    )


def validate_yaml_paths(yml_path: Path, repo_root: Path) -> list[Path]:
    missing: list[Path] = []
    with yml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for raw_value in _iter_path_values(data):
        if "$ASSETS_PATH" in raw_value or "${ASSETS_PATH}" in raw_value:
            missing.append(Path(raw_value))
            continue
        if not _looks_like_path(raw_value):
            continue
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.exists():
            # Ignore non-path labels in mixed config fields.
            if re.search(r"\.(urdf|srdf|ya?ml|glb|stl|dae)$", raw_value) or raw_value.startswith(('/', './', '../')):
                missing.append(candidate)
    return missing


def validate_embodiment(repo_root: Path, name: str) -> None:
    emb_dir = repo_root / "assets" / "embodiments" / name
    required = REQUIRED_BY_EMBODIMENT.get(name, [])
    missing = [emb_dir / rel for rel in required if not (emb_dir / rel).exists()]
    for yml_name in ("curobo.yml", "curobo_left.yml", "curobo_right.yml"):
        yml_path = emb_dir / yml_name
        if yml_path.exists():
            missing.extend(validate_yaml_paths(yml_path, repo_root))
    if missing:
        print_color(f"[embodiment-config] validation failed for {name}", YELLOW)
        for path in missing:
            print(f"  - missing: {path}")
        raise SystemExit(2)
    print_color(f"[embodiment-config] validated {name}", GREEN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="RMBench checkout root. Defaults to the parent of script/.")
    parser.add_argument(
        "--validate",
        nargs="*",
        default=DEFAULT_VALIDATE,
        help="Embodiment names to validate after generation. Use --no-validate to skip.",
    )
    parser.add_argument("--no-validate", action="store_true", help="Generate configs without validating required files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_args(args.repo_root)
    generate_configs(repo_root)
    apply_robot_poses(repo_root)
    apply_joint_limits(repo_root)
    if not args.no_validate:
        for name in args.validate:
            validate_embodiment(repo_root, name)
    print_color("[embodiment-config] complete", GREEN)


if __name__ == "__main__":
    main()
