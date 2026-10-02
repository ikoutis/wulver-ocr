"""Stage 2: a general VLM proofreads the specialist's output, under a gate.

The division of labour:
  * the READER (a small document-OCR specialist) is fast and pixel-faithful
    but has failure modes it cannot see in itself: repetition loops, silently
    truncated output, unbalanced LaTeX, a misread sub/superscript;
  * the REVIEWER (a larger general VLM) is slow and fluent — good at noticing
    that a transcription does not match the crop, dangerous because fluency
    means it will happily "improve" notation or paraphrase.

So the reviewer never rewrites a page. It sees one block at a time — the
image crop plus the reader's draft — and its proposal is accepted only if
  1. it introduces no validator flag the draft did not already have, and
  2. it changes at most ``max_change`` of the draft (1 - difflib ratio),
     unless the draft was already degenerate (empty / looping / truncated),
     in which case the reviewer is the fallback reader for that block.
Rejected proposals are kept in the block's history, so the gate's decisions
are auditable and its thresholds tunable from data (see design.md §Eval).

Which blocks are reviewed is a policy: every block with a validator flag, plus
every block whose type is in ``review_types`` (formulas by default — the
emphasis of this project, and the place where a single wrong subscript
matters).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .backend import (ChatClient, ServerError, Stopped, check_stop, image_part,
                      map_concurrent, text_part)
from .schema import Block, Page
from .validate import strip_math_delims, validate_block

DEGENERATE = {"empty", "repetition", "truncated"}

_COMMON_RULES = """Rules:
- Transcribe only what is visible in the image. Never add, complete, or summarise.
- Do not rename variables, reorder terms, or "improve" notation, even if it looks unusual.
- If the draft is already exactly right, return it unchanged."""

_PROMPTS = {
    "formula": """You are proofreading OCR output for a scanned research paper.
The image shows one display equation. The OCR draft of its LaTeX is:

<draft>
{draft}
</draft>
{issues}
Compare the draft with the image symbol by symbol: sub/superscripts, primes,
accents (hat, bar, tilde, dot), Greek letters, operators, delimiters and their
sizes, fractions, alignment, and any equation number (keep it as \\tag{{...}}).
{rules}
- Return the LaTeX body only, without $ or \\[ \\] delimiters.

Answer in exactly this format:
VERDICT: correct | fixed | unreadable
<latex>
...
</latex>""",
    "table": """You are proofreading OCR output for a scanned research paper.
The image shows one table. The OCR draft (HTML or Markdown) is:

<draft>
{draft}
</draft>
{issues}
Check every cell, merged cells (colspan/rowspan), and any math (keep it as $...$).
{rules}
- Return an HTML <table> if the table has merged cells, otherwise HTML or Markdown as in the draft.

Answer in exactly this format:
VERDICT: correct | fixed | unreadable
<table_out>
...
</table_out>""",
    "text": """You are proofreading OCR output for a scanned research paper.
The image shows one region of a page. The OCR draft (Markdown, math as $...$) is:

<draft>
{draft}
</draft>
{issues}
Check the wording, punctuation, and especially every piece of inline math.
{rules}
- Keep Markdown, with inline math as $...$.

Answer in exactly this format:
VERDICT: correct | fixed | unreadable
<text>
...
</text>""",
}
_OUT_TAG = {"formula": "latex", "table": "table_out", "text": "text"}


def prompt_kind(block_type: str) -> str:
    return block_type if block_type in ("formula", "table") else "text"


@dataclass
class ReviewPolicy:
    review_types: set = field(default_factory=lambda: {"formula"})
    flagged: bool = True
    max_change: float = 0.35          # for blocks the validators passed
    max_change_flagged: float = 0.6   # for flagged, non-degenerate blocks
    pad: float = 0.01                 # crop padding as a fraction of the page

    def wants(self, b: Block) -> bool:
        if b.type in ("figure", "header", "footer", "page_number"):
            return False
        return (self.flagged and bool(b.flags)) or b.type in self.review_types


def crop(img: Image.Image, bbox: Optional[list[float]], pad: float) -> Image.Image:
    if not bbox:
        return img
    w, h = img.size
    px, py = pad * w + 6, pad * h + 6
    x0, y0, x1, y1 = bbox
    box = (max(0, int(x0 * w - px)), max(0, int(y0 * h - py)),
           min(w, int(x1 * w + px)), min(h, int(y1 * h + py)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return img
    return img.crop(box)


def parse_reply(reply: str, kind: str) -> tuple[str, Optional[str]]:
    """-> (verdict, content or None). Tolerates missing closing tags."""
    m = re.search(r"VERDICT:\s*(correct|fixed|unreadable)", reply, re.I)
    verdict = m.group(1).lower() if m else "fixed"
    tag = _OUT_TAG[kind]
    m = re.search(rf"<{tag}>\s*\n?(.*?)\n?\s*(?:</{tag}>|$)", reply, re.S)
    content = m.group(1).strip() if m else None
    if content is not None and kind == "formula":
        content = strip_math_delims(content)
    return verdict, content


def change_fraction(a: str, b: str) -> float:
    a, b = " ".join(a.split()), " ".join(b.split())
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def gate(block: Block, proposal: str, policy: ReviewPolicy) -> tuple[bool, str]:
    """Decide whether the reviewer's proposal replaces the block content."""
    if not proposal.strip():
        return False, "empty proposal"
    trial = Block(type=block.type, content=proposal)
    new_flags = set(validate_block(trial))
    old_flags = set(block.flags)
    added = new_flags - old_flags
    if added:
        return False, f"introduces flags {sorted(added)}"
    if old_flags & DEGENERATE:
        return True, "draft degenerate; reviewer re-read accepted"
    limit = policy.max_change_flagged if old_flags else policy.max_change
    frac = change_fraction(block.content, proposal)
    if frac > limit:
        return False, f"change {frac:.2f} > limit {limit:.2f}"
    return True, f"change {frac:.2f}"


def review_block(client: ChatClient, page_img: Image.Image, block: Block,
                 policy: ReviewPolicy, model_tag: str) -> Block:
    """Ask the reviewer about one block; apply the gate; record provenance.

    Failure semantics (see backend.ChatClient): Stopped and ServerError
    propagate — the block was not reviewed and its page must not be saved as
    reviewed. Any other error (a 4xx RequestRejected, a parsing bug) is
    deterministic: it is recorded as this block's final review decision
    ("error"), so one bad request can never keep a page unreviewable."""
    check_stop()
    try:
        return _review_block(client, page_img, block, policy, model_tag)
    except (Stopped, ServerError):
        raise
    except Exception as e:      # noqa: BLE001 — recorded, not swallowed
        block.history.append({"source": model_tag, "decision": f"error: {e!r}"})
        block.meta["reviewed"] = "error"
        return block


def _review_block(client: ChatClient, page_img: Image.Image, block: Block,
                  policy: ReviewPolicy, model_tag: str) -> Block:
    kind = prompt_kind(block.type)
    issues = (f"\nAutomatic checks flagged: {', '.join(block.flags)}.\n"
              if block.flags else "\n")
    prompt = _PROMPTS[kind].format(draft=block.content, issues=issues,
                                   rules=_COMMON_RULES)
    reply = client.chat([image_part(crop(page_img, block.bbox, policy.pad)),
                         text_part(prompt)], max_tokens=4096)
    verdict, proposal = parse_reply(reply, kind)
    record = {"source": model_tag, "verdict": verdict}

    if verdict == "correct" or proposal is None or proposal == block.content:
        record["decision"] = "kept (reviewer agreed)" if verdict == "correct" \
            else "kept (no usable proposal)"
        block.history.append(record)
        block.meta["reviewed"] = "agreed" if verdict == "correct" else "no-proposal"
        return block
    if verdict == "unreadable":
        record["decision"] = "kept (reviewer: unreadable)"
        block.history.append(record)
        block.flags = sorted(set(block.flags) | {"unreadable"})
        block.meta["reviewed"] = "unreadable"
        return block

    ok, why = gate(block, proposal, policy)
    if ok:
        block.history.append({**record, "decision": f"accepted: {why}",
                              "previous": block.content,
                              "previous_source": block.source})
        block.content = proposal
        block.source = model_tag
        block.flags = validate_block(block)
        block.meta["reviewed"] = "edited"
    else:
        block.history.append({**record, "decision": f"rejected: {why}",
                              "proposal": proposal})
        block.meta["reviewed"] = "rejected"
    return block


def review_pages(client: ChatClient, pages: list[Page], policy: ReviewPolicy,
                 model_tag: str, workers: int = 16) -> set[int]:
    """Review every wanted block of these pages concurrently (library helper;
    the CLI pools blocks across a whole shard instead, see run_ocr).
    Returns the indices of pages left INCOMPLETE (a block hit Stopped or
    ServerError); only the other pages may be saved as reviewed."""
    images = {}
    for p in pages:
        with Image.open(p.image) as im:
            images[p.index] = im.convert("RGB")
    work = [(p, b) for p in pages for b in p.blocks if policy.wants(b)]
    results = map_concurrent(
        lambda pb: review_block(client, images[pb[0].index], pb[1], policy, model_tag),
        work, workers)
    incomplete = {p.index for (p, _), r in zip(work, results)
                  if isinstance(r, (Stopped, ServerError))}
    for p in pages:
        if p.index not in incomplete:
            p.stage = "review"
    return incomplete
