"""GPU-batched 6-DOF Starship upper-stage simulator."""

from starship.sim.config import SimConfig, SpawnConfig, VehicleConfig
from starship.sim.physics import CRASHED, FLYING, LANDED, OUT_OF_BOUNDS, TIMEOUT, TUMBLED
from starship.sim.simulator import TANK_ORDER, Simulator, Snapshot, TerminalSnapshot

__all__ = [
    "CRASHED",
    "FLYING",
    "LANDED",
    "OUT_OF_BOUNDS",
    "TANK_ORDER",
    "TIMEOUT",
    "TUMBLED",
    "SimConfig",
    "Simulator",
    "Snapshot",
    "SpawnConfig",
    "TerminalSnapshot",
    "VehicleConfig",
]
