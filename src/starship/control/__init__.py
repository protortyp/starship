"""Controllers that map simulator state to thruster actions: baselines and trained policies."""

from starship.control.heuristic import HeuristicController, IdleController, RandomController

CONTROLLER_NAMES = ("heuristic", "random", "idle", "policy")


def make_controller(name: str, sim, policy: str | None = None, seed: int = 0):
    if name == "policy":
        if policy is None:
            raise ValueError("the policy controller needs --policy <path to policy.pt>")
        from starship.control.policy import PolicyController

        return PolicyController(sim, policy)
    return {
        "heuristic": lambda: HeuristicController(sim.config),
        "random": lambda: RandomController(seed),
        "idle": IdleController,
    }[name]()


__all__ = [
    "CONTROLLER_NAMES",
    "HeuristicController",
    "IdleController",
    "RandomController",
    "make_controller",
]
