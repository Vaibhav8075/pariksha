"""Command-line entry point: `pariksha [paths]`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .explain import ExplainError, explain, settings
from .findings import severity_rank
from .rules import RULES, SEVERITIES
from .scan import ScanResult, scan

_COLOURS = {"high": "\033[31m", "medium": "\033[33m", "low": "\033[36m"}
_RESET, _DIM, _BOLD = "\033[0m", "\033[2m", "\033[1m"


def _paint(text: str, code: str, on: bool) -> str:
    return f"{code}{text}{_RESET}" if on else text


def _summary(result: ScanResult) -> str:
    counts = Counter(f.severity for f in result.findings)
    parts = ", ".join(f"{counts[s]} {s}" for s in SEVERITIES if counts[s])
    noun = "finding" if len(result.findings) == 1 else "findings"
    files = "file" if result.files == 1 else "files"
    return (f"{result.files} {files} scanned, {len(result.findings)} {noun}"
            + (f" ({parts})" if parts else ""))


def print_text(result: ScanResult, colour: bool, explanations: dict[int, str]) -> None:
    for i, f in enumerate(result.findings):
        sev = _paint(f.severity.upper().ljust(6), _COLOURS[f.severity], colour)
        print(f"{_paint(f.where, _BOLD, colour)}  {sev}  {f.rule}  {RULES[f.rule].title}")
        print(f"    {f.message}")
        if f.code:
            print(_paint(f"    > {f.code}", _DIM, colour))
        print(f"    fix: {RULES[f.rule].fix}")
        if i in explanations:
            print("    explanation (BharatCode):")
            for line in explanations[i].splitlines():
                print(f"      {line}")
        print()
    for path, reason in result.skipped:
        print(f"skipped {path}: {reason}", file=sys.stderr)
    print(_summary(result))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pariksha",
        description="Find evaluation mistakes (data leakage, test-set reuse, missing eval mode, "
                    "unseeded randomness) in machine-learning code and notebooks.")
    p.add_argument("paths", nargs="*", default=["."], help="files or folders to scan (default: .)")
    p.add_argument("--json", action="store_true", help="print findings as JSON")
    p.add_argument("--explain", action="store_true",
                   help="ask the BharatCode API to explain each finding (needs BHARATCODE_API_KEY)")
    p.add_argument("--max-explain", type=int, default=5, metavar="N",
                   help="explain at most N findings, highest severity first (default: 5)")
    p.add_argument("--select", metavar="RULES", help="only these rules, e.g. PX001,PX002")
    p.add_argument("--fail-on", choices=[*SEVERITIES, "none"], default="high",
                   help="exit with status 1 if a finding at or above this severity exists "
                        "(default: high)")
    p.add_argument("--list-rules", action="store_true", help="list the rules and exit")
    p.add_argument("--no-colour", "--no-color", action="store_true", help="plain output")
    p.add_argument("--version", action="version", version=f"pariksha {__version__}")
    return p


def _explain_all(result: ScanResult, limit: int) -> dict[int, str]:
    api_key, base_url, model = settings()
    if not api_key:
        print("--explain needs BHARATCODE_API_KEY (set it in the environment or a .env file); "
              "showing findings without explanations.", file=sys.stderr)
        return {}
    order = sorted(range(len(result.findings)),
                   key=lambda i: severity_rank(result.findings[i].severity))[:max(limit, 0)]
    out = {}
    for i in order:
        try:
            out[i] = explain(result.findings[i], api_key=api_key, base_url=base_url, model=model)
        except ExplainError as err:
            print(f"explanation skipped: {err}", file=sys.stderr)
            break
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Code lines can hold characters the Windows console encoding cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.list_rules:
        for rule in RULES.values():
            print(f"{rule.id}  {rule.title}\n       fix: {rule.fix}")
        return 0

    select = None
    if args.select:
        select = {r.strip().upper() for r in args.select.split(",") if r.strip()}
        unknown = select - RULES.keys()
        if unknown:
            print(f"unknown rule(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2

    missing = [p for p in args.paths if not Path(p).exists()]
    if missing:
        print(f"no such file or folder: {', '.join(missing)}", file=sys.stderr)
        return 2

    result = scan([Path(p) for p in args.paths], select)
    explanations = _explain_all(result, args.max_explain) if args.explain else {}

    if args.json:
        items = [f.to_dict() for f in result.findings]
        for i, text in explanations.items():
            items[i]["explanation"] = text
        json.dump({"version": __version__, "files_scanned": result.files, "findings": items,
                   "skipped": [{"file": p, "reason": r} for p, r in result.skipped]},
                  sys.stdout, indent=2)
        print()
    else:
        colour = sys.stdout.isatty() and not args.no_colour and "NO_COLOR" not in os.environ
        print_text(result, colour, explanations)

    if args.fail_on == "none":
        return 0
    threshold = severity_rank(args.fail_on)
    return int(any(severity_rank(f.severity) <= threshold for f in result.findings))


if __name__ == "__main__":
    sys.exit(main())
