from pathlib import Path

from starship.train.diagnose import promotion_lines, run_summary


def scalars(landed, critic):
    steps = [i * 100 for i in range(len(landed))]
    return {
        "train/iteration": [(t, i) for i, t in enumerate(steps)],
        "train/env/rate/landed": list(zip(steps, landed, strict=True)),
        "loss/critic": list(zip(steps, critic, strict=True)),
    }


def test_promotion_lines_flag_collapse_and_critic_spike():
    # Promotion at iteration 60; the first 10 iterations after it are ignored.
    landed = [0.95] * 60 + [0.0] * 10 + [0.3] * 40 + [0.9] * 100
    critic = [40.0] * 60 + [40.0, 4000.0] + [40.0] * 148
    lines = promotion_lines(scalars(landed, critic), [{"stage": 3, "env_steps": 6000}])
    assert lines == [
        "- WARN stage 3 transition: train landed 0.95 -> low 0.30, recovered in 50 it, "
        "critic loss peak 100x the prior median"
    ]


def test_promotion_lines_ignore_iterations_with_few_episodes():
    landed = [0.95] * 60 + [0.0] * 10 + [0.9] * 140
    s = scalars(landed, [40.0] * 210)
    s["train/env/episodes"] = [
        (t, 3.0 if i == 100 else 1000.0) for i, (t, _) in enumerate(s["loss/critic"])
    ]
    s["train/env/rate/landed"][100] = (10000, 0.0)
    lines = promotion_lines(s, [{"stage": 3, "env_steps": 6000}])
    assert "low 0.90" in lines[0]


def test_promotion_lines_quiet_for_smooth_transition():
    landed = [0.95] * 60 + [0.0] * 10 + [0.9] * 140
    lines = promotion_lines(scalars(landed, [40.0] * 210), [{"stage": 3, "env_steps": 6000}])
    assert lines == [
        "- stage 3 transition: train landed 0.95 -> low 0.90, recovered in 10 it, "
        "critic loss peak 1x the prior median"
    ]


def test_run_summary_warns_when_kl_stop_starves_updates():
    s = {"optim/updates": [(i, 1.0 if i % 2 else 20.0) for i in range(100)]}
    lines = run_summary(Path("run"), s, {})
    assert any(line.startswith("- WARN policy updates per iteration") for line in lines)
    s = {"optim/updates": [(i, 20.0) for i in range(100)]}
    lines = run_summary(Path("run"), s, {})
    assert "- policy updates per iteration (last 100): mean 20.0" in lines


def test_promotion_lines_without_promotions():
    assert promotion_lines(scalars([0.9] * 20, [40.0] * 20), []) == []
