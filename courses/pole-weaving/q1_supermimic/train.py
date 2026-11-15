"""Train the Q1 SuperMimic pole-weaving task."""

from __future__ import annotations

import argparse
from pathlib import Path

from .profiles import resolve_profile


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pole-weaving",), default="pole-weaving")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    profile = resolve_profile(args.profile, args.config)
    forwarded = ["--config", str(profile.config_path)]
    if args.resume is not None:
        forwarded.extend(("--resume-checkpoint", str(args.resume)))
    if args.check:
        forwarded.append("--preflight-only")

    from pole_task.training import main as pole_main

    return pole_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
