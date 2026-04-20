from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GifOption:
    gif_id: str
    path: Path
    description: str

    @property
    def alt_text(self) -> str:
        return self.description


class GifCatalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def list_options(self) -> list[GifOption]:
        if not self.root.exists() or not self.root.is_dir():
            return []

        options: list[GifOption] = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() != ".gif":
                continue
            gif_id = path.stem.strip().lower()
            if not gif_id:
                continue
            options.append(
                GifOption(
                    gif_id=gif_id,
                    path=path,
                    description=_describe_gif_id(gif_id),
                )
            )
        return options

    def get(self, gif_id: str | None) -> GifOption | None:
        if not gif_id:
            return None
        normalized = gif_id.strip().lower()
        for option in self.list_options():
            if option.gif_id == normalized:
                return option
        return None


def _describe_gif_id(gif_id: str) -> str:
    return " ".join(part for part in gif_id.replace("-", "_").split("_") if part)
