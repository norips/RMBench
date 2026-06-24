#!/usr/bin/env bash
set -euo pipefail

# In-place patches for installed `sapien` and `mplib`.
# Replaces the two `sed -i` calls from the original install script.
# Runs inside the pixi environment; the sed substitutions are self-idempotent
# (re-running is a no-op once applied, since the matched text no longer exists).

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION="$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
if [[ -f "$URDF_LOADER" ]]; then
    # open(..., "r") as f  ->  open(..., "r", encoding="utf-8") as f
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8")\3/g' "$URDF_LOADER"
    echo "  [done] $URDF_LOADER"
else
    echo "  [skip] $URDF_LOADER not found"
fi

echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION="$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib"
PLANNER="$MPLIB_LOCATION/planner.py"
if [[ -f "$PLANNER" ]]; then
    # drop the `or collide` term from the screw-plan failure check
    sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
    echo "  [done] $PLANNER"
else
    echo "  [skip] $PLANNER not found"
fi