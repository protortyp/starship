"""Webcast OCR descent telemetry; timestamps come from 30 fps frame names, not MissionTime."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Telemetry:
    t: np.ndarray  # seconds since the first descending 20 km reading
    altitude_m: np.ndarray
    speed_mps: np.ndarray
    repaired: np.ndarray  # (samples, 2): altitude / speed interpolated, corrected or rejected
    start_time: float = 0.0  # frame clock seconds
    burn_time: float | None = None  # frame clock seconds, inferred from final braking
    altitude_half_width: float = 500.0  # nearest-km quantization


def _number(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _clean(t: np.ndarray, values: np.ndarray, tolerance: float):
    valid = np.isfinite(values) & (values >= 0)
    if valid.sum() < 2:
        raise ValueError("telemetry channel needs at least two valid readings")
    filled = np.interp(t, t[valid], values[valid])
    # A one-second median rejects isolated OCR spikes without smoothing the km plateaus.
    width = max(1, round(0.5 / np.median(np.diff(t))))
    median = np.median(
        np.lib.stride_tricks.sliding_window_view(np.pad(filled, width, mode="edge"), 2 * width + 1),
        axis=1,
    )
    valid &= abs(filled - median) <= tolerance
    if valid.sum() < 2:
        raise ValueError("too few readings after OCR rejection")
    return np.interp(t, t[valid], values[valid]), ~valid


def load_telemetry(
    path: str | Path, *, corrected: bool = True, burn_margin: float = 2.5
) -> Telemetry:
    """Clean and select descent through 20 km, ending before the final braking episode.

    Repaired channels remain in the series for plotting but should be excluded from fitting.
    Burn onset is inferred using one-second speed differences below 3 km, within 40 seconds
    of the first near-zero speed; it is not a measured ignition time.
    """
    if burn_margin < 0:
        raise ValueError("burn_margin must be nonnegative")
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            frame = re.fullmatch(r"frame_(\d+)\.png", row["FileName"].strip())
            if frame is None:
                raise ValueError(f"invalid frame name: {row['FileName']!r}")
            rows.append(
                [int(frame[1]) / 30]
                + [
                    _number(row[f"Starship{field}_{source}"])
                    for source in ("Raw", "Corrected")
                    for field in ("Altitude", "Velocity")
                ]
            )
    if len(rows) < 3:
        raise ValueError("telemetry needs at least three frames")
    values = np.asarray(rows, dtype=float)
    values = values[np.argsort(values[:, 0], kind="stable")]
    _, indices = np.unique(values[:, 0], return_index=True)
    values = values[indices]
    if len(values) < 3:
        raise ValueError("telemetry needs at least three distinct frames")
    t = values[:, 0]
    source = values[:, 3:5] if corrected else values[:, 1:3]
    altitude, bad_h = _clean(t, source[:, 0], 1.1)
    speed, bad_v = _clean(t, source[:, 1] / 3.6, 15.0)
    repaired = np.column_stack((bad_h, bad_v))
    if corrected:
        repaired |= ~np.isfinite(values[:, 1:3]) | (source != values[:, 1:3])
    high = np.flatnonzero(altitude > 25)
    if not len(high):
        raise ValueError("no descent from above 25 km")
    starts = np.flatnonzero((np.arange(len(t)) > high[0]) & (altitude <= 20))
    if not len(starts):
        raise ValueError("no descending 20 km crossing")
    start = starts[0]
    if altitude[start] < 19.5 or repaired[start].any():
        raise ValueError("20 km crossing is missing or repaired; cannot establish spawn")
    low = np.flatnonzero((np.arange(len(t)) > start) & (altitude < 3) & (speed < 10))
    if not len(low):
        raise ValueError("no near-zero landing speed to locate the burn")
    near_zero = t[low[0]]
    clock = np.arange(max(t[start], near_zero - 40), near_zero, 1.0)
    v = np.interp(clock, t, speed)
    h = np.interp(clock, t, altitude)
    braking = (np.diff(v) < -3.0) & (v[:-1] > 30) & (h[:-1] < 3)
    onsets = np.flatnonzero(braking)
    if not len(onsets):
        raise ValueError("no final braking episode found")
    # Use the last contiguous strong-braking episode, ignoring earlier OCR dips.
    groups = np.split(onsets, np.flatnonzero(np.diff(onsets) > 1) + 1)
    sustained = [group for group in groups if len(group) >= 2]
    if not sustained:
        raise ValueError("no sustained final braking episode found")
    burn = clock[sustained[-1][0]]
    selected = (t >= t[start]) & (t <= burn - burn_margin)
    if selected.sum() < 2:
        raise ValueError("descent window is empty")
    return Telemetry(
        t[selected] - t[start],
        altitude[selected] * 1000,
        speed[selected],
        repaired[selected],
        start_time=t[start],
        burn_time=burn,
    )
