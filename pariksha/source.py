"""Load Python files and Jupyter notebooks into parseable source."""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules",
    "site-packages", ".ipynb_checkpoints", "build", "dist", ".tox", ".mypy_cache",
    ".pytest_cache", ".eggs",
}

# Cell magics whose body is not Python.
_FOREIGN_CELL_MAGICS = ("%%bash", "%%sh", "%%html", "%%javascript", "%%js", "%%latex",
                        "%%markdown", "%%writefile", "%%sql", "%%script", "%%perl", "%%ruby")


@dataclass
class Source:
    path: Path
    display: str
    text: str
    # Notebooks only: (cell number, first line of the cell in `text`) for each code cell.
    cells: list[tuple[int, int]] = field(default_factory=list)

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    def line(self, n: int) -> str:
        lines = self.lines
        return lines[n - 1] if 0 < n <= len(lines) else ""

    def cell_line(self, n: int) -> tuple[int, int]:
        """Notebook only: (cell number, line within that cell) for line n of `text`."""
        cell, first = self.cells[0]
        for c, f in self.cells:
            if f > n:
                break
            cell, first = c, f
        return cell, n - first + 1

    def where(self, n: int) -> str:
        if not self.cells:
            return f"{self.display}:{n}"
        cell, line = self.cell_line(n)
        return f"{self.display} [cell {cell}]:{line}"

    def describe_lines(self, message: str) -> str:
        """Rewrite "line N" in a message as "cell C line L" for notebooks."""
        if not self.cells:
            return message
        return re.sub(r"(?<!\w)line (\d+)(?!\d)",
                      lambda m: "cell {} line {}".format(*self.cell_line(int(m.group(1)))), message)

    def parse(self) -> ast.Module:
        return ast.parse(self.text, filename=str(self.path))


def _display(path: Path) -> str:
    try:
        return os.path.relpath(path).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return str(path)


def _notebook_cells(path: Path) -> list[tuple[int, list[str]]]:
    nb = json.loads(path.read_text(encoding="utf-8-sig"))
    cells = []
    for number, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        lines = text.splitlines()
        if lines and lines[0].strip().startswith(_FOREIGN_CELL_MAGICS):
            lines = [""] * len(lines)
        # Line magics and shell escapes are not Python; blank them so line numbers stay put.
        lines = ["" if ln.lstrip().startswith(("%", "!")) else ln for ln in lines]
        cells.append((number, lines))
    return cells


def load_notebook(path: Path) -> Source:
    cells = _notebook_cells(path)
    while True:
        text_lines: list[str] = []
        starts: list[tuple[int, int]] = []
        for number, lines in cells:
            starts.append((number, len(text_lines) + 1))
            text_lines.extend(lines or [""])
        src = Source(path, _display(path), "\n".join(text_lines) + "\n", starts)
        try:
            src.parse()
            return src
        except SyntaxError as err:
            # Blank the cell that does not parse (IPython-only syntax) and try again.
            bad = next((i for i, (_, first) in reversed(list(enumerate(starts)))
                        if err.lineno is not None and first <= err.lineno), None)
            if bad is None or not any(cells[bad][1]):
                raise
            cells[bad] = (cells[bad][0], [""] * len(cells[bad][1]))


def load(path: Path) -> Source:
    if path.suffix == ".ipynb":
        return load_notebook(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return Source(path, _display(path), text)


def discover(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        if p.is_file():
            found.append(p)
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for name in sorted(files):
                if name.endswith((".py", ".ipynb")):
                    found.append(Path(root) / name)
    return found
