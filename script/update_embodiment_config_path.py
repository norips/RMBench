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

DEFAULT_VALIDATE = ["franka-panda"]
REQUIRED_BY_EMBODIMENT = {
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
    if not args.no_validate:
        for name in args.validate:
            validate_embodiment(repo_root, name)
    print_color("[embodiment-config] complete", GREEN)


if __name__ == "__main__":
    main()
