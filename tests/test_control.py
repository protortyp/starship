from starship.control import HeuristicController
from starship.sim import LANDED, Simulator


def test_heuristic_lands_base_scenario():
    sim = Simulator(4, device="cpu")
    controller = HeuristicController(sim.config)
    for _ in range(sim.config.max_steps):
        sim.step(controller(sim.snapshot()))
    assert (sim.snapshot().status == LANDED).all()
