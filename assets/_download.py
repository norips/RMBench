#!/usr/bin/env python3
"""Download RMBench assets in a reproducible, idempotent way.

Embodiments are intentionally downloaded from RoboTwin2.0's `embodiments.zip`
archive.  RMBench task objects are a separate dataset namespace and are fetched
from `TianxingChen/RMBench`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download, snapshot_download

ROBOTWIN_REPO_ID = "TianxingChen/RoboTwin2.0"
ROBOTWIN_EMBODIMENTS_FILE = "embodiments.zip"
RMBENCH_REPO_ID = "TianxingChen/RMBench"

REQUIRED_FRANKA_FILES = [
    "config.yml",
    "panda.urdf",
    "panda.srdf",
    "curobo_tmp.yml",
    "collision_franka.yml",
    "franka_description/meshes/visual/link0.glb",
    "franka_description/meshes/visual/link7.glb",
    "franka_description/meshes/visual/camera_base.glb",
    "franka_description/meshes/visual/d435.dae",
    "franka_description/meshes/collision/link0.stl",
    "franka_description/meshes/collision/link7.stl",
    "franka_description/meshes/collision/hand.stl",
    "franka_description/meshes/collision/finger.stl",
]

# Minimal object set used by swap_blocks + the global imports that enumerate
# clutterable assets.  The default still downloads the full RMBench object tree;
# these paths are validated explicitly so a partial/mirrored install fails early.
REQUIRED_SWAP_BLOCK_OBJECT_FILES = [
    "objects/002_breadbasket/collision/base1.glb",
    "objects/002_breadbasket/visual/base1.glb",
    "objects/002_breadbasket/model_data1.json",
    "objects/005_button/10124/mobility.urdf",
    "objects/005_button/10124/model_data.json",
    "objects/cube/textured.obj",
    "objects/same.json",
    "objects/objaverse/list.json",
]


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent


def _safe_extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract `zip_path` into `dest_dir`, skipping macOS metadata safely."""
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            posix = PurePosixPath(name)
            if (
                name.endswith("/")
                or posix.parts[:1] == ("__MACOSX",)
                or any(part.startswith("._") for part in posix.parts)
                or any(part == ".DS_Store" for part in posix.parts)
            ):
                continue
            target = (dest_dir / Path(*posix.parts)).resolve()
            if not str(target).startswith(str(dest_dir) + os.sep):
                raise RuntimeError(f"Refusing to extract path outside assets dir: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _remove_appledouble_files(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()
    for path in root.rglob(".DS_Store"):
        if path.is_file():
            path.unlink()
    macosx = root / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)


def download_robotwin_embodiments(assets_dir: Path) -> Path:
    print(
        "[assets] downloading RoboTwin embodiments from "
        f"dataset {ROBOTWIN_REPO_ID}/{ROBOTWIN_EMBODIMENTS_FILE}"
    )
    archive = hf_hub_download(
        repo_id=ROBOTWIN_REPO_ID,
        repo_type="dataset",
        filename=ROBOTWIN_EMBODIMENTS_FILE,
        local_dir=assets_dir / ".cache" / "huggingface" / "robotwin2",
    )
    archive_path = Path(archive)
    print(f"[assets] extracting {archive_path} -> {assets_dir}")
    _safe_extract(archive_path, assets_dir)
    _remove_appledouble_files(assets_dir / "embodiments")
    patch_franka_config(assets_dir)
    return archive_path


def patch_franka_config(assets_dir: Path) -> None:
    """Keep RoboTwin's single-Panda embodiment explicit after extraction."""
    config_path = assets_dir / "embodiments" / "franka-panda" / "config.yml"
    if not config_path.exists():
        return
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("gripper_stiffnes:", "gripper_stiffness:")
    canonical_pose = (
        "robot_pose: [[0, -0.65, 0.75, 0.707, 0, 0, 0.707],\n"
        "             [0, -0.65, 0.75, 0.707, 0, 0, 0.707]]"
    )
    marker = "robot_pose:"
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        if lines[i].startswith(marker):
            out.extend(
                [
                    "# franka-panda is one physical SAPIEN articulation.  RMBench exposes both",
                    "# logical left/right slots for policy compatibility, but both slots alias this",
                    "# same base pose and the same joints.",
                    canonical_pose,
                ]
            )
            i += 1
            while i < len(lines) and lines[i].startswith(" "):
                i += 1
            replaced = True
            continue
        out.append(lines[i])
        i += 1
    if replaced:
        text = "\n".join(out) + "\n"
    # Keep current canonical local config explicit even when upstream formatting changes.
    text = text.replace("gripper_stiffnes:", "gripper_stiffness:")
    config_path.write_text(text, encoding="utf-8")


def download_rmbench_objects(assets_dir: Path, object_patterns: list[str]) -> None:
    print(
        "[assets] downloading RMBench object assets from "
        f"dataset {RMBENCH_REPO_ID} patterns={object_patterns}"
    )
    snapshot_download(
        repo_id=RMBENCH_REPO_ID,
        repo_type="dataset",
        allow_patterns=object_patterns,
        local_dir=assets_dir,
    )


def _missing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def validate_franka_files(assets_dir: Path) -> None:
    franka_dir = assets_dir / "embodiments" / "franka-panda"
    missing_franka = _missing([franka_dir / rel for rel in REQUIRED_FRANKA_FILES])
    missing_objects = _missing([assets_dir / rel for rel in REQUIRED_SWAP_BLOCK_OBJECT_FILES])
    missing = missing_franka + missing_objects
    if missing:
        print("[assets] missing required files after download:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        raise SystemExit(2)
    print(f"[assets] verified franka-panda embodiment at {franka_dir}")
    print("[assets] verified swap_blocks object assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects",
        choices=["full", "swap_blocks"],
        default=os.environ.get("RMBENCH_OBJECT_ASSET_SET", "full"),
        help=(
            "Object asset subset to download. 'full' mirrors assets/objects/** "
            "from the RMBench dataset; 'swap_blocks' downloads only the files "
            "needed by demo_clean_franka + swap_blocks. Environment: "
            "RMBENCH_OBJECT_ASSET_SET."
        ),
    )
    parser.add_argument(
        "--skip-objects",
        action="store_true",
        default=os.environ.get("RMBENCH_SKIP_OBJECT_ASSETS", "").lower() in {"1", "true", "yes", "on"},
        help="Only download/extract RoboTwin embodiments, then validate existing object files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_dir = _assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)

    download_robotwin_embodiments(assets_dir)

    if not args.skip_objects:
        if args.objects == "swap_blocks":
            object_patterns = REQUIRED_SWAP_BLOCK_OBJECT_FILES
        else:
            object_patterns = ["objects/**"]
        download_rmbench_objects(assets_dir, object_patterns)
    else:
        print("[assets] skipping RMBench object download by request")

    validate_franka_files(assets_dir)


if __name__ == "__main__":
    main()
