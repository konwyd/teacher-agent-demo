"""Small on-demand skill catalog backed by local Markdown files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSummary:
    name: str
    title: str
    description: str


class SkillCatalog:
    """Discover and safely load only skills inside the prototype directory."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(__file__).with_name("skills")
        self._paths = {path.stem: path for path in sorted(self._root.glob("*.md"))}

    def list(self) -> list[SkillSummary]:
        summaries: list[SkillSummary] = []
        for name, path in self._paths.items():
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            title = next(
                (line.removeprefix("#").strip() for line in lines if line.startswith("#")),
                name,
            )
            description = next(
                (
                    line
                    for line in lines
                    if line and not line.startswith("#") and not line.startswith("---")
                ),
                "",
            )
            summaries.append(
                SkillSummary(name=name, title=title, description=description)
            )
        return summaries

    def load(self, name: str) -> str:
        path = self._paths.get(name)
        if path is None:
            available = ", ".join(sorted(self._paths)) or "none"
            raise KeyError(f"unknown skill {name!r}; available: {available}")
        return path.read_text(encoding="utf-8")
