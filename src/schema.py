"""The page/block data model shared by every stage.

A reader (stage 1) turns a page image into an ordered list of Blocks; the
reviewer (stage 2) may replace a block's content, keeping the old version in
``history``; assembly (stage 3) turns blocks into Markdown. Every stage reads
and writes the same JSON, so any stage can be re-run on its own and a page's
provenance — which model wrote which block, what the validators flagged, which
edits were accepted or rejected — is always on disk next to the output.

Coordinates: ``bbox`` is [x0, y0, x1, y1] as FRACTIONS of the page image's
width/height (0..1), so it survives re-rendering at a different DPI. Readers
convert from whatever their model emits.

Content conventions by block type:
  formula          LaTeX body WITHOUT delimiters (assembly adds $$ ... $$)
  table            HTML (<table>...) or GitHub-flavoured Markdown
  figure           "" — the image crop lives in meta["image"], the generated
                   description in meta["description"]
  everything else  Markdown, inline math as $...$
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Optional

# The canonical block vocabulary. Reader adapters map their model's labels onto
# these; anything unmappable becomes "text" (kept) or "other" (kept, flagged).
BLOCK_TYPES = (
    "title",        # document title
    "heading",      # section header; meta["level"] = 1.. when known
    "text",         # body paragraph
    "list",         # list item(s)
    "formula",      # display equation
    "table",
    "figure",       # picture / plot / diagram
    "caption",      # figure or table caption
    "footnote",
    "header",       # running page header   (dropped by assembly)
    "footer",       # running page footer   (dropped by assembly)
    "page_number",  #                       (dropped by assembly)
    "code",         # code / pseudocode / algorithm
    "reference",    # bibliography entry
    "other",
)
DROPPED_TYPES = frozenset({"header", "footer", "page_number"})


@dataclass
class Block:
    type: str
    content: str = ""
    bbox: Optional[list[float]] = None
    source: str = ""                         # "<role>:<model>", e.g. "reader:dots.ocr"
    flags: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in BLOCK_TYPES:
            raise ValueError(f"unknown block type {self.type!r}")


@dataclass
class Page:
    doc_id: str
    index: int                               # 0-based page index in the document
    image: str                               # path of the rendered page image
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)
    reader: str = ""                         # model that produced the blocks
    stage: str = "read"                      # "read" | "review"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        d = dict(d)
        d["blocks"] = [Block(**b) for b in d.get("blocks", [])]
        return cls(**d)

    def save(self, path: str) -> None:
        """Atomic write: a killed job never leaves a half-written page."""
        atomic_write_text(path, self.to_json())

    @classmethod
    def load(cls, path: str) -> "Page":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# mkstemp creates files 0600; outputs on /project are meant to be group-
# readable, so apply the process umask as open() would. (Read once, at import,
# because os.umask can only be read by setting it.)
_UMASK = os.umask(0o022)
os.umask(_UMASK)


def atomic_write_text(path: str, text: str) -> None:
    """Write via a uniquely named temp file in the same directory, then
    rename: readers never see a partial file, and two writers of the same
    path never share a temp file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def page_stem(index: int) -> str:
    """p0001, p0002, ... (1-based in file names, matching how people count pages)."""
    return f"p{index + 1:04d}"
