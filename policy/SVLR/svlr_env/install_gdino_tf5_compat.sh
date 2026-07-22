#!/usr/bin/env bash
set -euo pipefail

# Install the GroundingDINO <-> transformers 5.x compatibility shim into the
# SVLR conda environment. This is required because SVLR's grounded_sam2 backend
# imports the vendored (transformers-4.x-era) GroundingDINO, while the SVLR env
# ships transformers 5.x (pinned indirectly by gradio 6.x / huggingface-hub 1.x).
#
# The shim restores ModuleUtilsMixin.get_head_mask and makes
# get_extended_attention_mask tolerate GroundingDINO's (mask, shape, device)
# call, without downgrading transformers and without editing SVLR or the
# Grounded-SAM-2 source tree.
#
# Usage:
#   bash install_gdino_tf5_compat.sh [/path/to/svlr/python]
# Defaults to the `svlr` conda env python.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_BIN="${1:-}"
if [ -z "${PY_BIN}" ]; then
  for cand in \
    "${CONDA_PREFIX:-}/bin/python" \
    "$HOME/miniconda3/envs/svlr/bin/python" \
    "$HOME/anaconda3/envs/svlr/bin/python"; do
    if [ -x "${cand}" ]; then PY_BIN="${cand}"; break; fi
  done
fi
if [ -z "${PY_BIN}" ] || [ ! -x "${PY_BIN}" ]; then
  echo "Could not find the svlr env python. Pass it explicitly: bash install_gdino_tf5_compat.sh /path/to/python" >&2
  exit 1
fi

SP="$("${PY_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
echo "[gdino-compat] target site-packages: ${SP}"

install -m 0644 "${SCRIPT_DIR}/_gdino_tf5_compat.py" "${SP}/_gdino_tf5_compat.py"
printf 'import _gdino_tf5_compat\n' > "${SP}/zz_gdino_tf5_compat.pth"

"${PY_BIN}" - <<'PYCHECK'
from transformers.modeling_utils import ModuleUtilsMixin
assert hasattr(ModuleUtilsMixin, "get_head_mask"), "get_head_mask shim not active"
assert getattr(ModuleUtilsMixin.get_extended_attention_mask, "_gdino_tf5_wrapped", False), "gext shim not active"
print("[gdino-compat] shim active in a fresh interpreter: OK")
PYCHECK

echo "[gdino-compat] installed."
