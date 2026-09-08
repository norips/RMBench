#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python assets/_download.py "$@"

echo "[assets] configuring embodiment Curobo paths ..."
python script/update_embodiment_config_path.py --repo-root "${REPO_ROOT}" --validate aloha-agilex franka-panda
