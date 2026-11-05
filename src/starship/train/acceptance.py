"""Broad held-out acceptance for the exact single-salto landing policy.

The final curriculum stage makes touchdown count only when the rocket lands inside every safety
limit after exactly one flip and zero re-flips.  This command repeats that strict stage across
independent spawn streams and fails unless every stream clears the requested landing-rate floor.
"""

import argparse
import sys

from starship.train.diagnose import find_run, probe_section

DEFAULT_SEEDS = (101, 211, 307, 401, 503)


def stream_passes(stats: dict[str, float], minimum_landed: float) -> bool:
    """Whether one held-out stream meets both rate and exact attitude invariants."""
    return (
        stats.get("rate/landed", 0.0) >= minimum_landed
        and stats.get("landed/attitude/flips") == 1.0
        and stats.get("landed/attitude/reflips") == 0.0
    )


def main() -> None:
    from starship.train.conf import load_run_config
    from starship.train.curriculum import Curriculum

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="experiment name under logs/ or a run directory")
    parser.add_argument("--episodes", type=int, default=1024, help="episodes per held-out seed")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="independent held-out spawn seeds",
    )
    parser.add_argument("--minimum-landed", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0.0 <= args.minimum_landed <= 1.0:
        parser.error("--minimum-landed must be between zero and one")

    run = find_run(args.run)
    cfg = load_run_config(run)
    if not cfg.curriculum.enabled:
        raise SystemExit("single-salto acceptance requires an enabled curriculum")
    curriculum = Curriculum(cfg, cfg.curriculum)
    final_stage = len(curriculum.stages) - 1
    final_sim, _ = curriculum.configs(final_stage)
    if not final_sim.require_single_loop_before_landing:
        raise SystemExit("final curriculum stage does not enforce the exact single-salto invariant")

    rows = []
    for seed in args.seeds:
        report: dict = {}
        probe_section(run, report, args.episodes, args.device, final_stage, seed)
        stats = report["probe"]["stats"]
        rows.append((seed, stats, stream_passes(stats, args.minimum_landed)))

    print("| Seed | Landed | Flips | Reflips | Result |")
    print("|---:|---:|---:|---:|:---:|")
    for seed, stats, passed in rows:
        print(
            f"| {seed} | {stats.get('rate/landed', 0.0):.1%} | "
            f"{stats.get('landed/attitude/flips', float('nan')):.2f} | "
            f"{stats.get('landed/attitude/reflips', float('nan')):.2f} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    passed = all(row[2] for row in rows)
    print(
        f"\n{'PASS' if passed else 'FAIL'}: {len(args.seeds) * args.episodes:,} held-out "
        f"episodes; every seed must land at least {args.minimum_landed:.0%} with exactly one "
        "flip and zero re-flips."
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
