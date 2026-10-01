"""Figures: crop every figure block to a PNG, and (optionally) have the
reviewer VLM describe it — with a machine-usable structure where one exists.

The reader localises figures but cannot say what they show. The reviewer is
asked to classify each figure and, by kind, to emit:
  graph               an edge list ("u -- v", "u -> v", "u -- v : w") —
                      graph drawings in papers are usually small and fully
                      labelled, and an edge list is what a reader wants
  diagram             Mermaid (renders natively on GitHub / Obsidian)
  commutative_diagram tikz-cd
  plot                series, axes, and legible key values
The crop is always kept and linked; the description is an addition, never a
replacement, and is marked as generated in the Markdown.
"""

from __future__ import annotations

import os
import re

from PIL import Image

from .backend import ChatClient, image_part, map_concurrent, text_part
from .review import crop
from .schema import Block, Page, page_stem

FIGURE_KINDS = ("plot", "graph", "diagram", "commutative_diagram",
                "table_image", "algorithm", "photo", "other")
_FENCE = {"graph": "text", "diagram": "mermaid",
          "commutative_diagram": "latex", "plot": "text"}

DESCRIBE_PROMPT = """This image is a figure from a research paper.
Caption (may be empty): "{caption}"

1. Classify it as exactly one of: plot, graph, diagram, commutative_diagram, table_image, algorithm, photo, other.
   ("graph" = a drawing of vertices and edges; "diagram" = flowchart / block / architecture diagram.)
2. Describe what it shows in 2-6 factual sentences. For plots: axes, units, series, and the main trend.
   Do not speculate beyond what is visible and stated in the caption.
3. Give a structured transcription when the kind allows it, otherwise leave it empty:
   - graph: one edge per line as "u -- v" (undirected) or "u -> v" (directed), using the visible vertex
     labels (name unlabelled vertices v1, v2, ... left-to-right, top-to-bottom); add " : w" for edge weights.
   - diagram: Mermaid flowchart code.
   - commutative_diagram: tikz-cd code (the body of \\begin{{tikzcd}} ... \\end{{tikzcd}}).
   - plot: one line per series with approximate key values, only if legible.

Answer in exactly this format:
KIND: <kind>
<description>
...
</description>
<structure>
...
</structure>"""


def caption_for(blocks: list[Block], i: int) -> str:
    """Nearest caption block after (else before) figure i, within 2 blocks."""
    for j in list(range(i + 1, min(len(blocks), i + 3))) + \
             list(range(i - 1, max(-1, i - 3), -1)):
        if blocks[j].type == "caption":
            return blocks[j].content.strip()
    return ""


def parse_description(reply: str) -> dict:
    m = re.search(r"KIND:\s*([a-z_]+)", reply, re.I)
    kind = m.group(1).lower() if m else "other"
    kind = kind if kind in FIGURE_KINDS else "other"
    d = re.search(r"<description>\s*(.*?)\s*(?:</description>|<structure>|$)", reply, re.S)
    s = re.search(r"<structure>\s*(.*?)\s*(?:</structure>|$)", reply, re.S)
    structure = s.group(1).strip() if s else ""
    structure = re.sub(r"^```[a-z-]*\n|\n?```$", "", structure).strip()
    return {"kind": kind, "summary": d.group(1).strip() if d else "",
            "structure": structure}


def format_description(info: dict) -> str:
    out = f"**Kind:** {info['kind']}. {info['summary']}".strip()
    if info.get("structure"):
        fence = _FENCE.get(info["kind"], "text")
        out += f"\n\n```{fence}\n{info['structure']}\n```"
    return out


def crop_figures(pages: list[Page], doc_dir: str, pad: float = 0.005) -> None:
    """Save each figure block's crop under <doc_dir>/figures/ and link it."""
    fig_dir = os.path.join(doc_dir, "figures")
    for p in pages:
        figs = [(k, b) for k, b in enumerate(p.blocks) if b.type == "figure" and b.bbox]
        if not figs:
            continue
        os.makedirs(fig_dir, exist_ok=True)
        with Image.open(p.image) as im:
            img = im.convert("RGB")
        for k, b in figs:
            name = f"{page_stem(p.index)}_b{k:02d}.png"
            crop(img, b.bbox, pad).save(os.path.join(fig_dir, name))
            b.meta["image"] = f"figures/{name}"
            b.meta.setdefault("alt", f"Figure (page {p.index + 1})")


def describe_figures(client: ChatClient, pages: list[Page], model_tag: str,
                     workers: int = 8) -> None:
    """Fill meta['description'] / meta['kind'] for every cropped figure."""
    work = []
    for p in pages:
        for k, b in enumerate(p.blocks):
            if b.type == "figure" and b.meta.get("image") and "description" not in b.meta:
                work.append((p, k, b))

    def one(item):
        p, k, b = item
        path = os.path.join(os.path.dirname(os.path.dirname(p.image)), b.meta["image"])
        with Image.open(path) as im:
            fig = im.convert("RGB")
        prompt = DESCRIBE_PROMPT.format(caption=caption_for(p.blocks, k).replace('"', "'"))
        info = parse_description(client.chat([image_part(fig), text_part(prompt)],
                                             max_tokens=2048))
        b.meta["kind"] = info["kind"]
        b.meta["description"] = format_description(info)
        b.meta["description_source"] = model_tag

    for (p, k, b), r in zip(work, map_concurrent(one, work, workers)):
        if isinstance(r, Exception):
            b.history.append({"source": model_tag, "decision": f"describe error: {r!r}"})
