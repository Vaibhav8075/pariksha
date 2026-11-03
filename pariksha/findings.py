from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .rules import RULES, SEVERITIES

if TYPE_CHECKING:
    from .source import Source


@dataclass
class Finding:
    rule: str
    severity: str
    line: int
    message: str
    source: Source | None = None

    @property
    def where(self) -> str:
        return self.source.where(self.line) if self.source else f"line {self.line}"

    @property
    def code(self) -> str:
        return self.source.line(self.line).strip() if self.source else ""

    def snippet(self, context: int = 6) -> str:
        if self.source is None:
            return ""
        lines = self.source.lines
        lo, hi = max(1, self.line - context), min(len(lines), self.line + context)
        return "\n".join(f"{'>' if n == self.line else ' '} {n:4d} | {lines[n - 1]}"
                         for n in range(lo, hi + 1))

    def sort_key(self) -> tuple:
        return (self.source.display if self.source else "", self.line, self.rule)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "title": RULES[self.rule].title,
            "severity": self.severity,
            "file": self.source.display if self.source else None,
            "line": self.line,
            "location": self.where,
            "message": self.message,
            "code": self.code,
            "fix": RULES[self.rule].fix,
        }


def severity_rank(severity: str) -> int:
    return SEVERITIES.index(severity)
