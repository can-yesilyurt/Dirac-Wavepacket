"""
dirac_wavepacket.__main__ — CLI entry point.

Usage
-----
  python -m dirac_wavepacket <config.yaml> [options]

Options
-------
  --output   -o   Output directory          (default: ./results)
  --no-anim       Skip GIF compilation
  --fps           Animation frames/second   (default: 15)
  --verbose  -v   Verbose progress output
  --check         Validate config and print summary, then exit
  --auto-grid     Auto-compute domain/grid from physics (overrides YAML grid)
  --no-detect     Disable flux detectors and auto-stop
  --recommend     Show auto-sizing recommendations and exit
"""

import argparse
import sys
from pathlib import Path


BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  dirac-wavepacket — 2D tilted Dirac wavepacket simulator  ║
  ║   Split-operator FFT  |  Flux-based T/R detection           ║
  ╚══════════════════════════════════════════════════════════════╝
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dirac_wavepacket",
        description="Simulate ballistic transport of a Dirac fermion wavepacket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m dirac_wavepacket configs/klein_tunneling.yaml -o ./results
  python -m dirac_wavepacket configs/oblique_30deg.yaml   -o ./results_30 --auto-grid
  python -m dirac_wavepacket configs/oblique_30deg.yaml   --recommend
  python -m dirac_wavepacket configs/free_propagation.yaml --check
        """,
    )
    parser.add_argument(
        "config",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        metavar="DIR",
        help="Output directory for frames, animation, and plots. (default: ./results)",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Skip compiling frames into a GIF animation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        metavar="N",
        help="GIF animation frame rate. (default: 15)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print step-by-step progress.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config and print simulation summary, then exit.",
    )
    parser.add_argument(
        "--auto-grid",
        action="store_true",
        help="Auto-compute optimal Lx, Ly, Nx, Ny from physics parameters.",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable flux detectors and auto-stop (original behaviour).",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Show auto-sizing recommendations for this config, then exit.",
    )
    return parser


def main(argv=None):
    print(BANNER)
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── load config ───────────────────────────────────────────────────────────
    from .config import load_config, config_summary
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR loading config: {e}", file=sys.stderr)
        sys.exit(1)

    cfg.output_dir = args.output

    # ── apply CLI overrides ───────────────────────────────────────────────────
    if args.auto_grid:
        cfg.detector.auto_grid = True
    if args.no_detect:
        cfg.detector.enabled = False
        cfg.detector.auto_stop = False
        cfg.detector.auto_source = False

    # ── recommend mode ────────────────────────────────────────────────────────
    if args.recommend:
        from .auto import print_auto_summary
        print(config_summary(cfg))
        print_auto_summary(cfg)
        sys.exit(0)

    # ── check mode ────────────────────────────────────────────────────────────
    if args.check:
        from .auto import validate_grid
        print(config_summary(cfg))
        warnings = validate_grid(cfg)
        if warnings:
            for w in warnings:
                print(f"  ⚠ {w}")
        else:
            print("  ✓ All grid checks passed.")
        print("  Config OK — exiting (--check mode).")
        sys.exit(0)

    # ── run ───────────────────────────────────────────────────────────────────
    from .simulation import run_simulation
    results = run_simulation(
        cfg=cfg,
        make_animation=not args.no_anim,
        fps=args.fps,
        verbose=True,
    )

    # Print final T/R if detectors were active
    if results.get("T") is not None:
        print(f"\n  Final:  T = {results['T']:.6f}   R = {results['R']:.6f}")


if __name__ == "__main__":
    main()
