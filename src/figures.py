"""Figures: crop every figure block to a PNG, and (optionally) have the
reviewer VLM describe it — with a machine-usable structure where one exists.

The reader localises figures but cannot say what they show. The reviewer is
asked to classify each figure and, by kind, to emit:
  graph               two marked versions of the same graph:
                        * Markdown (simple) — vertex list + edge list
                        * TikZ — a canonical tikzpicture redrawing the
                          vertices (at their drawn positions) and edges
                      The model writes only the TikZ (see tikz.py); it is
                      parsed back, the vertex/edge lists go into
                      meta["graph"], and the Markdown version is rendered
                      from them, so the two cannot disagree. A picture that
                      does not parse (or draws edges to undeclared vertices)
                      is sent back to the reviewer once with the problems
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

from .backend import (ChatClient, ServerError, Stopped, check_stop, image_part,
                      map_concurrent, text_part)
from .review import crop
from .schema import Block, Page, page_stem
from .tikz import check_tikzcd, graph_markdown, normalise, parse_graph

FIGURE_KINDS = ("plot", "graph", "diagram", "commutative_diagram",
                "table_image", "algorithm", "photo", "other")
_FENCE = {"graph": "latex", "diagram": "mermaid",
          "commutative_diagram": "latex", "plot": "text"}

# Built with str.replace, not str.format: the TikZ example is full of braces.
DESCRIBE_PROMPT = r"""This image is a figure from a research paper.
Caption (may be empty): "<<CAPTION>>"

1. Classify it as exactly one of: plot, graph, diagram, commutative_diagram, table_image, algorithm, photo, other.
   ("graph" = a drawing of vertices and edges; "diagram" = flowchart / block / architecture diagram.)
2. Describe what it shows in 2-6 factual sentences. For plots: axes, units, series, and the main trend.
   Do not speculate beyond what is visible and stated in the caption.
3. Give a structured transcription when the kind allows it, otherwise leave it empty:
   - graph: TikZ that redraws it, in exactly this form:
       \begin{tikzpicture}
         \node[circle, draw, inner sep=1.5pt] (a) at (0,2) {$a$};
         \node[circle, draw, inner sep=1.5pt] (b) at (1.5,2) {$b$};
         \node[circle, draw, inner sep=1.5pt] (v1) at (1.5,0) {};
         \draw (a) -- (b);
         \draw[->] (b) -- (v1);
         \draw (a) -- node[midway, auto] {$3$} (v1);
         \draw (b) to[bend left] (v1);
         \draw (v1) to[loop below] (v1);
       \end{tikzpicture}
     One \node per vertex: a short name, its position copied from the drawing (x to the right, y up,
     roughly 0-10), and its label exactly as shown ({} if unlabelled; name unlabelled vertices
     v1, v2, ... left-to-right, top-to-bottom). One \draw per edge: -- for straight edges, [->] if
     directed, to[bend left/right] for curved ones, node[midway, auto] {...} for an edge weight or
     label. Include every vertex and every edge, and nothing that is not drawn. No \foreach.
   - diagram: Mermaid flowchart code.
   - commutative_diagram: tikz-cd code (\begin{tikzcd} ... \end{tikzcd}).
   - plot: one line per series with approximate key values, only if legible.

Answer in exactly this format:
KIND: <kind>
<description>
...
</description>
<structure>
...
</structure>"""

REPAIR_SUFFIX = r"""

Your previous answer's TikZ could not be checked:
<<PROBLEMS>>
Previous TikZ:
<<PREVIOUS>>
Answer again, in the same format, with the TikZ fixed (same form as the example)."""


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


def check_structure(info: dict) -> tuple[dict, list[str], list[str]]:
    """Normalise and check a graph / commutative-diagram transcription.
    -> (info with normalised structure (+ parsed graph), flags, problems)."""
    if not info.get("structure"):
        return info, [], []
    if info["kind"] == "graph":
        info = dict(info, structure=normalise(info["structure"]))
        graph, flags, problems = parse_graph(info["structure"])
        if graph is not None:
            info["graph"] = graph
        return info, flags, problems
    if info["kind"] == "commutative_diagram":
        info = dict(info, structure=normalise(info["structure"], "tikzcd"))
        flags, problems = check_tikzcd(info["structure"])
        return info, flags, problems
    return info, [], []


def format_description(info: dict) -> str:
    out = f"**Kind:** {info['kind']}. {info['summary']}".strip()
    if info["kind"] == "graph" and info.get("structure"):
        # Two marked versions of the same graph: simple Markdown (derived from
        # the parsed TikZ, so they cannot disagree) and the TikZ itself.
        md = graph_markdown(info["graph"]) if info.get("graph") and info["graph"]["nodes"] \
            else "*(not available: the TikZ below did not parse)*"
        out += (f"\n\n**Graph — Markdown (simple):** {md}"
                f"\n\n**Graph — TikZ:**\n\n```latex\n{info['structure']}\n```")
    elif info.get("structure"):
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


def needs_description(b: Block) -> bool:
    return b.type == "figure" and bool(b.meta.get("image")) and "description" not in b.meta


def describe_figure(client: ChatClient, page: Page, k: int, model_tag: str,
                    repair: int = 1) -> Block:
    """Describe figure block ``k`` of ``page`` (meta['description'], ['kind'],
    and ['graph'] for graph drawings). A graph or commutative diagram whose
    TikZ fails the checks is sent back ``repair`` times with the problems;
    the last answer is kept either way, and remaining problems become flags.

    Same failure contract as review.review_block: Stopped and ServerError
    propagate (the page is not saved); any other error is recorded on the
    block as final, so it cannot keep the page from completing."""
    check_stop()
    b = page.blocks[k]
    try:
        _describe(client, page, k, b, model_tag, repair)
    except (Stopped, ServerError):
        raise
    except Exception as e:      # noqa: BLE001 — recorded, not swallowed
        b.history.append({"source": model_tag, "decision": f"describe error: {e!r}"})
        b.meta["description_error"] = repr(e)[:300]
    return b


def _describe(client, page, k, b, model_tag, repair):
    path = os.path.join(os.path.dirname(os.path.dirname(page.image)), b.meta["image"])
    with Image.open(path) as im:
        fig = im.convert("RGB")
    base = DESCRIBE_PROMPT.replace("<<CAPTION>>", caption_for(page.blocks, k).replace('"', "'"))
    prompt = base
    for attempt in range(repair + 1):
        if attempt:
            check_stop()
        info = parse_description(client.chat([image_part(fig), text_part(prompt)],
                                             max_tokens=4096))
        info, flags, problems = check_structure(info)
        if not flags or attempt == repair:
            break
        prompt = base + REPAIR_SUFFIX.replace("<<PROBLEMS>>", "\n".join(
            f"- {x}" for x in problems)).replace("<<PREVIOUS>>", info["structure"])
    b.meta["kind"] = info["kind"]
    b.meta["description"] = format_description(info)
    b.meta["description_source"] = model_tag
    b.meta["describe_attempts"] = attempt + 1
    if "graph" in info:
        b.meta["graph"] = info["graph"]
    b.flags = sorted(set(b.flags) | set(flags))


def describe_figures(client: ChatClient, pages: list[Page], model_tag: str,
                     workers: int = 8, repair: int = 1) -> set[int]:
    """Describe every figure of these pages that has a crop and no
    description yet (library helper; the CLI pools figures with block reviews
    across the shard). Returns the indices of pages left INCOMPLETE."""
    work = [(p, k) for p in pages for k, b in enumerate(p.blocks) if needs_description(b)]
    results = map_concurrent(lambda pk: describe_figure(client, pk[0], pk[1], model_tag,
                                                        repair), work, workers)
    return {p.index for (p, _), r in zip(work, results)
            if isinstance(r, (Stopped, ServerError))}
