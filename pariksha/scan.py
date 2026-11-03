"""Run every check over a set of files and collect the findings."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .checks import run_file_checks, torch_seed_facts
from .findings import Finding
from .source import Source, discover, load

_IGNORE = re.compile(r"#\s*pariksha:\s*ignore(?:\[([A-Z0-9,\s]+)\])?", re.I)


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    files: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _suppressed(finding: Finding) -> bool:
    match = _IGNORE.search(finding.source.line(finding.line)) if finding.source else None
    if not match:
        return False
    rules = match.group(1)
    return rules is None or finding.rule in {r.strip().upper() for r in rules.split(",")}


def scan(paths: list[Path], select: set[str] | None = None) -> ScanResult:
    result = ScanResult()
    first_backward: tuple[Source, int] | None = None
    seeded = False
    for path in discover(paths):
        try:
            src = load(path)
            tree = src.parse()
        except (SyntaxError, ValueError, OSError) as err:
            result.skipped.append((str(path), f"{type(err).__name__}: {err}"))
            continue
        result.files += 1
        for finding in run_file_checks(tree, src.text):
            finding.source = src
            finding.message = src.describe_lines(finding.message)
            result.findings.append(finding)
        backward, has_seed = torch_seed_facts(tree)
        seeded = seeded or has_seed
        if backward is not None and first_backward is None:
            first_backward = (src, backward)

    if first_backward is not None and not seeded:
        src, line = first_backward
        result.findings.append(Finding(
            "PX004", "low", line,
            "PyTorch training runs here, but no file in the project sets a seed "
            "(torch.manual_seed, seed_everything, ...), so weight initialisation and shuffling "
            "differ on every run.", src))

    result.findings = sorted(
        (f for f in result.findings
         if not _suppressed(f) and (select is None or f.rule in select)),
        key=Finding.sort_key)
    return result
