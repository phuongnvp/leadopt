from __future__ import annotations

import argparse
import sys
from typing import Optional


def _build_top_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leadopt",
        description="leadopt: academic molecular optimization (CLI umbrella)",
        add_help=True,
    )
    ap.add_argument(
        "command",
        nargs="?",
        help="Subcommand: sanity | run | train | generate | beam",
    )
    ap.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the subcommand (use `leadopt <cmd> --help`).",
    )
    return ap


def _print_top_help() -> None:
    # Use argparse to keep formatting consistent.
    ap = argparse.ArgumentParser(
        prog="leadopt",
        description="leadopt: academic molecular optimization (CLI umbrella)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("command", nargs="?", help="Subcommand to run")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ep = (
        "Subcommands:\n"
        "  sanity    Verify installation and (optionally) run a tiny rollout\n"
        "  run       Run a preset on a single SMILES (random rollout)\n"
        "  train     Train PPO for lead optimization\n"
        "  generate  Generate molecules from a trained PPO policy\n"
        "  beam      Generate molecules with deterministic beam search\n\n"
        "Use:\n"
        "  leadopt <command> --help\n"
    )
    ap.epilog = ep
    ap.print_help()


def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Global help / no-args: print top-level help
    if not argv or argv[0] in ("-h", "--help"):
        _print_top_help()
        return

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "sanity":
        from leadopt.cli.sanity import main as _sanity

        _sanity(rest)
        return
    if cmd == "run":
        from leadopt.cli.run import main as _run

        _run(rest)
        return
    if cmd == "train":
        from leadopt.cli.train import main as _train

        _train(rest)
        return
    if cmd == "generate":
        from leadopt.cli.generate import main as _gen

        _gen(rest)
        return
    if cmd == "beam":
        from leadopt.cli.beam import main as _beam

        _beam(rest)
        return

    raise SystemExit(
        f"Unknown command: {cmd}\nRun `leadopt --help` to see available subcommands."
    )


if __name__ == "__main__":
    main()
