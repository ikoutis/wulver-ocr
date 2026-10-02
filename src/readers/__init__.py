"""Stage-1 reader adapters: one per document-OCR model.

Every adapter turns one page image into an ordered list of schema.Blocks
(canonical types, bbox as page fractions, content per the schema's
conventions), so everything downstream is model-independent. Adding a model
means adding a module here and a line to READERS.
"""

from __future__ import annotations

from .base import Reader
from .chandra import ChandraReader
from .dots import DotsReader
from .markdown import OlmOCRReader, PageMarkdownReader

READERS: dict[str, type[Reader]] = {
    ChandraReader.name: ChandraReader,
    DotsReader.name: DotsReader,
    PageMarkdownReader.name: PageMarkdownReader,
    OlmOCRReader.name: OlmOCRReader,
}


def get_reader(name: str) -> type[Reader]:
    try:
        return READERS[name]
    except KeyError:
        raise SystemExit(f"unknown reader {name!r}; choose from {sorted(READERS)}")
