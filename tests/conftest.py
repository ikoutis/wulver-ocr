"""Shared fixtures: synthetic documents and a scriptable fake model server.

No GPU, model, or network is needed anywhere in the test suite: model servers
are replaced by httpx.MockTransport handlers that answer the OpenAI chat
format, so the real client, readers, reviewer, and gate code all run.
"""

from __future__ import annotations

import json
import os
import sys

import httpx
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend import ChatClient  # noqa: E402


def make_page(text: str = "Hello", size=(850, 1100)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((80, 80), text, fill="black")
    d.rectangle((100, 600, 500, 900), outline="black")   # a "figure"
    return img


@pytest.fixture
def pdf_path(tmp_path):
    """A 2-page PDF written by Pillow."""
    path = tmp_path / "paper one.pdf"
    pages = [make_page("Page one"), make_page("Page two")]
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=100)
    return str(path)


def chat_reply(text: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}]}


class FakeServer:
    """Routes each chat request to ``responder(prompt_text, n_images) -> str``
    (or -> (str, finish_reason)), and records requests for assertions."""

    def __init__(self, responder, model="fake-model"):
        self.responder = responder
        self.model = model
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": self.model}]})
        body = json.loads(request.content)
        self.requests.append(body)
        content = body["messages"][-1]["content"]
        if isinstance(content, str):
            prompt, n_img = content, 0
        else:
            prompt = "\n".join(p["text"] for p in content if p["type"] == "text")
            n_img = sum(p["type"] == "image_url" for p in content)
        out = self.responder(prompt, n_img)
        text, finish = out if isinstance(out, tuple) else (out, "stop")
        return httpx.Response(200, json=chat_reply(text, finish))

    def client(self) -> ChatClient:
        return ChatClient("http://fake:8000", transport=httpx.MockTransport(self.handler),
                          max_retries=0)
