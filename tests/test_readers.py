import json

from PIL import Image

from conftest import FakeServer
from src.backend import TRUNCATION_MARKER
from src.readers.base import normalise_bbox, smart_resize
from src.readers.dots import DotsReader, parse_elements
from src.readers.markdown import PageMarkdownReader, split_markdown, strip_front_matter


class TestSmartResize:
    def test_multiples_and_bounds(self):
        for h, w in [(2200, 1700), (1100, 850), (50, 40), (9000, 7000)]:
            nh, nw = smart_resize(h, w, 28, 3136, 11289600)
            assert nh % 28 == 0 and nw % 28 == 0
            assert 3136 <= nh * nw <= 11289600
            if min(h, w) > 500:        # tiny inputs drift by a rounding step
                assert abs(nh / nw - h / w) < 0.05

    def test_bbox_normalisation(self):
        assert normalise_bbox([10, 20, 110, 220], 200, 400) == [0.05, 0.05, 0.55, 0.55]
        assert normalise_bbox([110, 220, 10, 20], 200, 400) == [0.05, 0.05, 0.55, 0.55]
        assert normalise_bbox([0, 0, 0, 10], 100, 100) is None
        assert normalise_bbox(None, 100, 100) is None


ELEMENTS = [
    {"bbox": [10, 10, 500, 40], "category": "Page-header", "text": "J. Graph Theory"},
    {"bbox": [50, 60, 800, 120], "category": "Title", "text": "Spectral Sparsifiers"},
    {"bbox": [50, 130, 800, 160], "category": "Section-header", "text": "## 1 Introduction"},
    {"bbox": [50, 170, 800, 300], "category": "Text", "text": "Let $G=(V,E)$ be a graph."},
    {"bbox": [200, 310, 600, 350], "category": "Formula", "text": "$$L = D - A$$"},
    {"bbox": [100, 400, 700, 800], "category": "Picture"},
    {"bbox": [100, 810, 700, 840], "category": "Caption", "text": "Figure 1: A graph."},
]


class TestDots:
    def test_parse_complete(self):
        els, complete = parse_elements(json.dumps(ELEMENTS))
        assert complete and len(els) == len(ELEMENTS)

    def test_parse_truncated_keeps_complete_objects(self):
        s = json.dumps(ELEMENTS)
        cut = s[: s.index('"Picture"') + 5]          # cut inside the 6th object
        els, complete = parse_elements(cut)
        assert not complete and len(els) == 5

    def test_parse_code_fence(self):
        els, complete = parse_elements("```json\n" + json.dumps(ELEMENTS[:2]) + "\n```")
        assert complete and len(els) == 2

    def test_read_maps_types_and_boxes(self):
        srv = FakeServer(lambda prompt, n: json.dumps(ELEMENTS))
        blocks = DotsReader(srv.client()).read(Image.new("RGB", (840, 1092), "white"))
        assert [b.type for b in blocks] == ["header", "title", "heading", "text",
                                           "formula", "figure", "caption"]
        assert blocks[2].content == "1 Introduction" and blocks[2].meta["level"] == 1
        assert blocks[4].content == "L = D - A"            # delimiters stripped
        assert all(0 <= v <= 1 for b in blocks for v in b.bbox)
        # 840x1092 is already a multiple of 28 inside the pixel bounds: no resize
        assert blocks[1].bbox[0] == 50 / 840
        assert srv.requests[0]["messages"][0]["content"][1]["text"].startswith(
            "<|img|><|imgpad|><|endofimg|>")

    def test_read_truncated_adds_tail_block(self):
        # cut inside the Picture: the five complete elements are kept as they
        # are (no marker on the intact formula); a tail block covers the rest
        s = json.dumps(ELEMENTS)
        srv = FakeServer(lambda p, n: (s[: s.index('"Picture"')], "length"))
        blocks = DotsReader(srv.client()).read(Image.new("RGB", (840, 1092)))
        assert [b.type for b in blocks] == ["header", "title", "heading", "text",
                                           "formula", "text"]
        assert blocks[4].content == "L = D - A"
        assert not any(TRUNCATION_MARKER.strip() in b.content for b in blocks)
        tail = blocks[-1]
        assert tail.content == "" and tail.meta == {"truncated_tail": True}
        assert tail.bbox == [0.0, 350 / 1092, 1.0, 1.0]

    def test_retry_uses_sampling(self):
        srv = FakeServer(lambda p, n: "[]")
        r = DotsReader(srv.client())
        r.read(Image.new("RGB", (840, 1092)), attempt=1)
        body = srv.requests[0]
        assert body["temperature"] > 0 and body["repetition_penalty"] > 1


PAGE_MD = r"""---
primary_language: en
is_table: False
---
# A Title

Some text with $x^2$ inline
continuing here.

$$
\sum_{i} a_i = 1 \tag{1}
$$

<table>
<tr><td>a</td><td>b</td></tr>

<tr><td>c</td><td>d</td></tr>
</table>

| x | y |
|---|---|
| 1 | 2 |

- item one
- item two

![figure](figure)
"""


class TestMarkdownReader:
    def test_front_matter(self):
        body, meta = strip_front_matter(PAGE_MD)
        assert meta["primary_language"] == "en" and body.startswith("# A Title")

    def test_split(self):
        body, _ = strip_front_matter(PAGE_MD)
        blocks = split_markdown(body)
        assert [b.type for b in blocks] == ["heading", "text", "formula", "table",
                                           "table", "list", "figure"]
        assert blocks[2].content == r"\sum_{i} a_i = 1 \tag{1}"
        assert "</table>" in blocks[3].content       # spans the blank line

    def test_read(self):
        srv = FakeServer(lambda p, n: PAGE_MD)
        blocks = PageMarkdownReader(srv.client()).read(Image.new("RGB", (1700, 2200)))
        assert len(blocks) == 7 and all(b.bbox is None for b in blocks)
