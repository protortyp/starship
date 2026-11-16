"""Summarize two landing_history_probe outputs; numerical differences are not causal proof."""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
args = parser.parse_args()
provenance = {
    name: json.loads((path / "provenance.json").read_text())
    for name, path in [("reference", args.reference), ("candidate", args.candidate)]
}
for key in ["device", "torch", "warp", "envs_per_stage", "steps", "stages"]:
    if provenance["reference"][key] != provenance["candidate"][key]:
        raise ValueError(f"Probe settings differ: {key}")
results = {}
for stage in range(provenance["reference"]["stages"]):
    with (
        np.load(args.reference / f"stage-{stage}.npz") as left,
        np.load(args.candidate / f"stage-{stage}.npz") as right,
    ):
        if set(left.files) != set(right.files):
            raise ValueError(f"Recorded fields differ at stage {stage}")
        rows = {}
        for name in left.files:
            a, b = left[name], right[name]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Invalid arrays at stage {stage}: {name}")
            rows[name] = {
                "equal": bool(np.array_equal(a, b)),
                "max_abs_diff": float(np.max(np.abs(a.astype(float) - b.astype(float)))),
                "different_values": int(np.count_nonzero(a != b)),
            }
        results[str(stage)] = rows
print(json.dumps({"provenance": provenance, "results": results}, indent=2))
