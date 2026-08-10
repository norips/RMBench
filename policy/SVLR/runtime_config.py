import ast

import numpy as np


def apply_initial_homestate_override(args, value):
    """Override initial arm joints without modifying the embodiment asset."""

    if value is None:
        return
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                "initial_homestate must be a list of joint positions"
            ) from exc

    qpos = np.asarray(value, dtype=np.float64)
    if qpos.ndim != 1 or qpos.size == 0 or not np.isfinite(qpos).all():
        raise ValueError(
            "initial_homestate must be a finite one-dimensional joint vector"
        )

    for config_key in ("left_embodiment_config", "right_embodiment_config"):
        config = args[config_key]
        homestates = config.get("homestate")
        if not isinstance(homestates, list) or not homestates:
            raise ValueError(f"{config_key} has no homestate to override")
        for homestate in homestates:
            if len(homestate) != qpos.size:
                raise ValueError(
                    f"initial_homestate has {qpos.size} joints, but "
                    f"{config_key} expects {len(homestate)}"
                )
        config["homestate"] = [qpos.tolist() for _ in homestates]

    print(f"[SVLR eval] initial_homestate override: {qpos.tolist()}")
