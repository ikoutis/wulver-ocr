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
A proposal that differs from the draft only in whitespace counts as
agreement: the reader's text and provenance stay.
An empty draft (such as the tail block a reader adds for output lost at a
max_tokens cut) has nothing to proofread, so the reviewer is asked to
transcribe its crop instead; "correct" then means there is nothing to
transcribe, and an answer that is no transcription (a bare page number, a
note such as "N/A") is rejected. An accepted tail is marked
meta["tail_recovered"]. The truncation marker is never shown to the
reviewer, and a proposal containing it is never accepted.

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

from .backend import (TRUNCATION_MARKER, ChatClient, ServerError, Stopped, check_stop,
                      image_part, map_concurrent, text_part)
from .schema import Block, Page
from .validate import outside_math, strip_math_delims, validate_block

DEGENERATE = {"empty", "repetition", "truncated"}
# check_latex runs KaTeX only on structurally sound LaTeX, so a draft with one
# of these flags was never KaTeX-checked (see gate).
_STRUCTURAL = {"latex_braces", "latex_env", "latex_leftright", "latex_delims"}
_MARK = TRUNCATION_MARKER.strip()       # "<<TRUNCATED>>", with or without its newline
# The Markdown escapes and HTML entities readers write into text (htmlmd),
# each with the bare form that renders as markup instead. They are not in
# the image, so a reviewer may drop them: a*b then renders as emphasis and
# List<String> as an HTML tag. (Moving escaped text into math is no drop.)
_ESCAPES = {r"\*": r"(?<!\\)\*", r"\_": r"(?<!\\)_", r"\`": r"(?<!\\)`",
            "&lt;": r"<(?=[A-Za-z/!?])", "&amp;": r"&(?!amp;|lt;)(?=#?\w+;)",
            r"\#": r"(?m)^ {0,3}#", r"\-": r"(?m)^ {0,3}-(?=\s|$)",
            r"\+": r"(?m)^ {0,3}\+(?=\s|$)", r"\>": r"(?m)^ {0,3}>",
            r"\.": r"(?m)^ {0,3}\d{1,9}\.(?=\s|$)",
            "\\\n": r"(?<!\\)\n"}         # hard line break (htmlmd.HARD_BREAK, from <br>)
# Answers to "transcribe this region" that are not a transcription: a bare
# number (the page number at the foot of a tail crop), a placeholder, or a
# one-line note about the crop. (Answers without a letter or digit are
# caught too.)
_PLACEHOLDER = re.compile(
    r"[\W_]*(?:\d+|n/?a|none|empty|blank|nothing(?: else)?(?: to transcribe)?)[\W_]*"
    r"|[(\[][^()\[\]\n]*\b(?:transcri\w*|nothing|blank|empty|illegible|page numbers?)\b"
    r"[^()\[\]\n]*[)\]]", re.I)

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
- Keep the draft's Markdown escapes and HTML entities (&lt;, &amp;) exactly as they are.

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
- Keep the draft's Markdown escapes (\\*, \\_, \\$, \\#) and HTML entities (&lt;, &amp;)
  exactly as they are: they are not in the image, but they make the text render as printed.

Answer in exactly this format:
VERDICT: correct | fixed | unreadable
<text>
...
</text>""",
}
_OUT_TAG = {"formula": "latex", "table": "table_out", "text": "text"}

# For an empty draft: the reviewer is the fallback reader of the crop.
_TRANSCRIBE = """You are transcribing part of a scanned research paper.
The OCR reader produced no text for the region of the page shown in the image.
Transcribe {what}.
Rules:
- Transcribe only what is visible in the image. Never add, complete, or summarise.
- Leave out running headers, footers, and page numbers.
- {form}
- Answer VERDICT: fixed with the transcription. If there is nothing else to transcribe,
  answer VERDICT: correct with nothing inside the tags.

Answer in exactly this format:
VERDICT: fixed | correct | unreadable
<{tag}>
(the transcription, or nothing)
</{tag}>"""
_TRANSCRIBE_KIND = {
    "formula": ("the display equation it shows",
                "Return the LaTeX body only, without $ or \\[ \\] delimiters; "
                "keep any equation number as \\tag{...}."),
    "table": ("the table it shows",
              "Return an HTML <table> if the table has merged cells, otherwise "
              "Markdown; keep any math as $...$."),
    "text": ("everything it shows, in reading order",
             "Write Markdown, with inline math as $...$ and display math as "
             "$$...$$ on lines of their own."),
}


def prompt_kind(block_type: str) -> str:
    return block_type if block_type in ("formula", "table") else "text"


def _draft(content: str) -> str:
    """The block content as the reviewer sees it: without the truncation marker."""
    return content.replace(TRUNCATION_MARKER, "").replace(_MARK, "").strip()


def _recheck(b: Block) -> None:
    """KaTeX was down when the block was read (latex_unchecked): check it
    again now, so its real flags decide whether and how it is reviewed."""
    b.flags = sorted((set(b.flags) - {"latex_unchecked"}) | set(validate_block(b)))


@dataclass
class ReviewPolicy:
    review_types: set = field(default_factory=lambda: {"formula"})
    flagged: bool = True
    max_change: float = 0.35          # for blocks the validators passed
    max_change_flagged: float = 0.6   # for flagged, non-degenerate blocks
    pad: float = 0.01                 # crop padding as a fraction of the page

    def wants(self, b: Block) -> bool:
        """Whether to review this block. A block flagged latex_unchecked is
        re-checked first (its flags are updated in place): KaTeX may work now."""
        if b.type in ("figure", "header", "footer", "page_number"):
            return False
        if "page_failed" in b.flags:    # a failed page's placeholder: re-read, not reviewed
            return False
        if not b.bbox and not _draft(b.content):
            # An empty block of a box-less reader (markdown/olmocr): its region
            # is unknown, and from a whole-page crop the reviewer would
            # re-transcribe what was kept (a truncated tail) or some other
            # equation (an empty $$ $$). Leave it flagged for report.json.
            return False
        if "latex_unchecked" in b.flags:
            _recheck(b)
        # A latex_unchecked that remains is no reason to review on its own:
        # it says only that KaTeX is still down.
        flagged = bool(set(b.flags) - {"latex_unchecked"})
        return (self.flagged and flagged) or b.type in self.review_types


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


_FENCE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.S)


def _unfence(s: str) -> str:
    """'```latex\\nx\\n```' -> 'x': one code fence around a whole answer."""
    m = _FENCE.fullmatch(s.strip())
    return m.group(1).strip() if m and "```" not in m.group(1) else s


def parse_reply(reply: str, kind: str, draft: str = "") -> tuple[str, Optional[str]]:
    """-> (verdict, content or None). Tolerates missing closing tags, a
    verdict in Markdown emphasis ("**VERDICT:** fixed"), and an answer in a
    code fence, inside the tags or in place of them: the fence is dropped,
    unless the draft itself is one fenced block."""
    m = re.search(r"VERDICT\W*:\W*(correct|fixed|unreadable)", reply, re.I)
    verdict = m.group(1).lower() if m else "fixed"
    tag = _OUT_TAG[kind]
    m = re.search(rf"<{tag}>\s*\n?(.*?)\n?\s*(?:</{tag}>|$)", reply, re.S)
    if m:
        content = m.group(1).strip()
    else:
        fences = list(_FENCE.finditer(reply))
        content = fences[-1].group(0).strip() if fences else None
    if content is not None and _unfence(draft) == draft:
        content = _unfence(content)
    if content is not None and kind == "formula":
        content = strip_math_delims(content)
    return verdict, content


def change_fraction(a: str, b: str) -> float:
    a, b = " ".join(a.split()), " ".join(b.split())
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def same_text(a: str, b: str, kind: str) -> bool:
    """Equal up to whitespace that does not matter: in LaTeX, all of it
    except between two letters ("\\in S" is not "\\inS") and after a
    backslash (a control space); in Markdown, all but the line structure
    (list items, table rows, paragraphs)."""
    def norm(s: str) -> str:
        if kind == "formula":
            return re.sub(r"(?<![A-Za-z\\]) |(?<!\\) (?![A-Za-z])", "", " ".join(s.split()))
        lines = "\n".join(" ".join(ln.split()) for ln in s.strip().splitlines())
        return re.sub(r"\n{3,}", "\n\n", lines)
    return norm(a) == norm(b)


def dropped_escapes(draft: str, proposal: str) -> list[str]:
    """The draft's escapes that the proposal turned into markup: fewer of
    the escape and more of its bare form, outside math."""
    d, p = outside_math(draft), outside_math(proposal)
    return [esc for esc, bare in _ESCAPES.items()
            if p.count(esc) < d.count(esc)
            and len(re.findall(bare, p)) > len(re.findall(bare, d))]


def _placeholder(proposal: str) -> bool:
    return (not re.search(r"[^\W_]", proposal)
            or _PLACEHOLDER.fullmatch(proposal.strip()) is not None)


def gate(block: Block, proposal: str, policy: ReviewPolicy) -> tuple[bool, str]:
    """Decide whether the reviewer's proposal replaces the block content."""
    if not proposal.strip():
        return False, "empty proposal"
    if _MARK in proposal:           # the reviewer's reply was cut off, or echoed the marker
        return False, "truncated proposal"
    trial = Block(type=block.type, content=proposal)
    new_flags = set(validate_block(trial))
    old_flags = set(block.flags)
    added = new_flags - old_flags
    if old_flags & _STRUCTURAL:
        # The draft's KaTeX status was never computed, so a proposal that
        # fixes its structure does not "add" a KaTeX flag.
        added -= {"latex_katex", "latex_unchecked"}
    if added:
        return False, f"introduces flags {sorted(added)}"
    if old_flags & DEGENERATE:
        if _placeholder(proposal):
            return False, "not a transcription"
        return True, "draft degenerate; reviewer re-read accepted"
    draft = _draft(block.content)
    if prompt_kind(block.type) != "formula":
        lost = dropped_escapes(draft, proposal)
        if lost:
            return False, f"drops Markdown escapes {lost}"
    # latex_unchecked says nothing against the draft: it keeps the tight bound
    limit = (policy.max_change_flagged if old_flags - {"latex_unchecked"}
             else policy.max_change)
    frac = change_fraction(draft, proposal)
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
    if "latex_unchecked" in block.flags:
        # so the reviewer is told the draft's real problems, and the gate
        # compares against them
        _recheck(block)
    kind = prompt_kind(block.type)
    draft = _draft(block.content)
    if draft:
        issues = (f"\nAutomatic checks flagged: {', '.join(block.flags)}.\n"
                  if block.flags else "\n")
        prompt = _PROMPTS[kind].format(draft=draft, issues=issues, rules=_COMMON_RULES)
    else:
        what, form = _TRANSCRIBE_KIND[kind]
        prompt = _TRANSCRIBE.format(what=what, form=form, tag=_OUT_TAG[kind])
    reply = client.chat([image_part(crop(page_img, block.bbox, policy.pad)),
                         text_part(prompt)], max_tokens=4096)
    verdict, proposal = parse_reply(reply, kind, draft)
    record = {"source": model_tag, "verdict": verdict}

    if verdict == "unreadable":     # whatever came with it: a human should look
        record["decision"] = "kept (reviewer: unreadable)"
        block.history.append(record)
        block.flags = sorted(set(block.flags) | {"unreadable"})
        block.meta["reviewed"] = "unreadable"
        return block
    if proposal is None and verdict != "correct":
        record["decision"] = "kept (no usable proposal)"
        block.history.append(record)
        block.meta["reviewed"] = "no-proposal"
        return block
    # "correct" keeps the draft whatever came with it (for an empty draft:
    # there is nothing to transcribe), and so does a proposal equal to it.
    if verdict == "correct" or same_text(draft, proposal, kind):
        record["decision"] = ("kept (reviewer agreed)" if draft
                              else "kept (reviewer: nothing to transcribe)")
        if proposal and proposal != draft:
            record["proposal"] = proposal
        block.history.append(record)
        block.meta["reviewed"] = "agreed"
        return block

    ok, why = gate(block, proposal, policy)
    if ok:
        block.history.append({**record, "decision": f"accepted: {why}",
                              "previous": block.content,
                              "previous_source": block.source})
        block.content = proposal
        block.source = model_tag
        block.meta["reviewed"] = "edited"       # first: see validate_block
        block.flags = validate_block(block)
        if block.meta.get("truncated_tail"):
            block.meta["tail_recovered"] = True
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
