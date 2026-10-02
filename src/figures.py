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
                      is sent back to the reviewer once with the problems.
                      If it still fails, the figure is flagged (tikz_*), the
                      Markdown version says it is unavailable, and the
                      partial parse is kept as meta["graph_partial"]. An
                      answer with no TikZ at all is sent back the same way
                      (tikz_missing).
  diagram             Mermaid (renders natively on GitHub / Obsidian)
  commutative_diagram tikz-cd
  plot                series, axes, and legible key values
The crop is always kept and linked; the description is an addition, never a
replacement, and is marked as generated in the Markdown. An answer cut off
at the length limit is flagged "description_truncated" and its structure is
left out (a graph's TikZ is sent back once first, like a failed check).
"""

from __future__ import annotations

import os
import re
from typing import Sequence

from PIL import Image

from .backend import (TRUNCATION_MARKER, ChatClient, ServerError, Stopped, check_stop,
                      image_part, map_concurrent, text_part)
from .review import crop
from .schema import Block, Page, page_stem
from .tikz import check_tikzcd, graph_markdown, normalise, parse_graph

FIGURE_KINDS = ("plot", "graph", "diagram", "commutative_diagram",
                "table_image", "algorithm", "photo", "other")
_FENCE = {"graph": "latex", "diagram": "mermaid",
          "commutative_diagram": "latex", "plot": "text"}
_TIKZ_KINDS = ("graph", "commutative_diagram")    # checked; sent back on failure
_CUT_OFF = ("the answer was cut off at the length limit: write the TikZ compactly, "
            "one line per vertex and edge")
_MISSING = "no {} was given: write it inside <structure> ... </structure>"
_CODE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.S)

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
    """-> {kind, summary, structure} (+ truncated=True for a reply cut off at
    the length limit; the marker itself is removed). Tolerates Markdown
    around the label and value (**KIND:** graph, KIND: `commutative diagram`)
    and a structure in a code fence after the description instead of in its
    tags (as review.parse_reply does)."""
    truncated = TRUNCATION_MARKER.strip() in reply
    reply = reply.replace(TRUNCATION_MARKER.strip(), "")
    m = re.search(r"\bKIND\W*:\W*([a-z_ ]+)", reply, re.I)
    kind = re.sub(r"\s+", "_", m.group(1).strip().lower()) if m else "other"
    kind = next((k for k in FIGURE_KINDS                 # "graph drawing" -> graph
                 if kind == k or kind.startswith(k + "_")), "other")
    d = (re.search(r"<description>\s*(.*?)\s*</description>", reply, re.S)
         or re.search(r"<description>\s*(.*?)\s*(?:<structure>|(?=\n```)|$)", reply, re.S))
    s = re.search(r"<structure>\s*(.*?)\s*(?:</structure>|$)", reply, re.S)
    if s:
        structure = s.group(1).strip()
    elif kind in _FENCE:                    # a kind that has a structure
        fences = _CODE.findall(reply[d.end():] if d else reply)
        structure = fences[-1].strip() if fences else ""
    else:
        structure = ""
    structure = re.sub(r"^```[a-z-]*\n|\n?```$", "", structure).strip()
    info = {"kind": kind, "summary": d.group(1).strip() if d else "",
            "structure": structure}
    if truncated:
        info["truncated"] = True
    return info


def check_structure(info: dict) -> tuple[dict, list[str], list[str]]:
    """Normalise and check a graph / commutative-diagram transcription.
    -> (info with normalised structure (+ parsed graph), flags, problems).
    A cut-off answer fails the check for every kind (description_truncated),
    and so does a graph or commutative diagram with no structure at all
    (tikz_missing)."""
    flags, problems = (["description_truncated"], [_CUT_OFF]) if info.get("truncated") \
        else ([], [])
    if not info.get("structure"):
        if info["kind"] in _TIKZ_KINDS and not flags:
            what = "TikZ" if info["kind"] == "graph" else "tikz-cd code"
            return info, ["tikz_missing"], [_MISSING.format(what)]
        return info, flags, problems
    if info["kind"] == "graph":
        info = dict(info, structure=normalise(info["structure"]))
        graph, more, why = parse_graph(info["structure"])
        if graph is not None:
            info["graph"] = graph
        return info, sorted(set(flags + more)), problems + why
    if info["kind"] == "commutative_diagram":
        info = dict(info, structure=normalise(info["structure"], "tikzcd"))
        more, why = check_tikzcd(info["structure"])
        return info, sorted(set(flags + more)), problems + why
    return info, flags, problems


def format_description(info: dict, problems: Sequence[str] = ()) -> str:
    """The Markdown description. ``problems`` are what the final check found
    (check_structure): with any, a graph's Markdown version says it is
    unavailable instead of rendering a partial parse that would contradict
    the TikZ shown below it, and a graph given without TikZ says so. A
    cut-off answer's structure is left out."""
    out = f"**Kind:** {info['kind']}. {info['summary']}".strip()
    if info.get("truncated"):
        return out + ("\n\n*(cut off at the length limit: the description may be "
                      "incomplete, and any structure is left out)*")
    if info["kind"] == "graph" and info.get("structure"):
        # Two marked versions of the same graph: simple Markdown (derived from
        # the parsed TikZ, so they cannot disagree) and the TikZ itself.
        graph = info.get("graph")
        if graph and graph["nodes"] and not problems:
            md = graph_markdown(graph)
        else:
            why = "; ".join("`{}`".format(" ".join(p.replace("`", "'").split()))
                            for p in problems) or "it did not parse"
            md = f"*(not available: the TikZ below failed the checks: {why})*"
        out += (f"\n\n**Graph — Markdown (simple):** {md}"
                f"\n\n**Graph — TikZ:**\n\n```latex\n{info['structure']}\n```")
    elif info["kind"] == "graph":
        out += "\n\n**Graph — Markdown (simple):** *(not available: no TikZ was given)*"
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
    TikZ fails the checks (or whose answer was cut off) is sent back
    ``repair`` times with the problems. The last answer is kept either way;
    remaining problems become flags, and a graph that still fails gets no
    Markdown version and keeps its partial parse as meta['graph_partial'].

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
        if not flags or info["kind"] not in _TIKZ_KINDS or attempt == repair:
            break
        prompt = base + REPAIR_SUFFIX.replace("<<PROBLEMS>>", "\n".join(
            f"- {x}" for x in problems)).replace("<<PREVIOUS>>", info["structure"] or "(none)")
    b.meta["kind"] = info["kind"]
    b.meta["description"] = format_description(info, problems)
    b.meta["description_source"] = model_tag
    b.meta["describe_attempts"] = attempt + 1
    if "graph" in info:
        # Only a graph that passed every check is data to score (design §7).
        b.meta["graph_partial" if problems else "graph"] = info["graph"]
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
