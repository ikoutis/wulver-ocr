"""A minimal client for OpenAI-compatible chat endpoints (vLLM, SGLang, ...).

Both models in the pipeline are served the same way — `vllm serve <model>` on
the GPU node, the pipeline talking to it over localhost HTTP — so this file is
the only place that knows the wire format. Keeping the pipeline on plain HTTP
(no vLLM import) means its environment stays light, the serving environment
can be upgraded or swapped for a container independently, and the same code
runs against a server on a Wulver GPU node, a laptop, or a tunnel.

vLLM batches concurrent requests internally, so throughput comes from issuing
many requests at once (see ``map_concurrent``), not from batching here.
"""

from __future__ import annotations

import base64
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

import httpx
from PIL import Image


def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


class ChatClient:
    """Chat completions against one served model.

    ``model`` must be the name the server registered (``--served-model-name``
    or the HF repo id); ``None`` asks the server and uses its first model.
    """

    def __init__(self, base_url: str, model: Optional[str] = None,
                 api_key: str = "EMPTY", timeout: float = 600.0,
                 max_retries: int = 3,
                 transport: Optional[httpx.BaseTransport] = None,
                 default_extra: Optional[dict] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self._http = httpx.Client(timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})
        self.max_retries = max_retries
        # Merged into every request body, e.g. a reasoning model's switch
        # {"chat_template_kwargs": {"enable_thinking": false}} (see profiles/).
        self.default_extra = dict(default_extra or {})
        self.model = model or self._first_model()

    def _first_model(self) -> str:
        r = self._http.get(f"{self.base_url}/models")
        r.raise_for_status()
        return r.json()["data"][0]["id"]

    def chat(self, content: list[dict] | str, *, system: Optional[str] = None,
             max_tokens: int = 4096, temperature: float = 0.0,
             extra: Optional[dict] = None) -> str:
        """One user turn (text and/or images) -> the assistant's text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature,
                **self.default_extra}
        if extra:
            body.update(extra)
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", json=body)
                if r.status_code < 500:
                    r.raise_for_status()
                    choice = r.json()["choices"][0]
                    text = strip_thinking(choice["message"].get("content") or "")
                    if choice.get("finish_reason") == "length":
                        # Truncated output is a validator signal, not an error:
                        # the caller sees it via the marker and flags the block.
                        text += TRUNCATION_MARKER
                    return text
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except (httpx.TransportError, httpx.TimeoutException) as e:
                err = repr(e)
            if attempt == self.max_retries:
                raise RuntimeError(f"chat request failed after retries: {err}")
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")


TRUNCATION_MARKER = "\n<<TRUNCATED>>"


def strip_thinking(text: str) -> str:
    """Drop <think>…</think> reasoning a model emits inline (when the server
    runs without a reasoning parser). An unterminated <think> means the
    budget ran out mid-thought: nothing after it is an answer."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    return text.split("<think>", 1)[0] if "<think>" in text else text


def image_part(img: Image.Image) -> dict:
    return {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def map_concurrent(fn: Callable, items: Iterable, workers: int) -> list:
    """Run fn over items with a thread pool, preserving order. Exceptions are
    returned in place of results so one bad page never sinks a shard."""
    items = list(items)
    if workers <= 1:
        out = []
        for it in items:
            try:
                out.append(fn(it))
            except Exception as e:      # noqa: BLE001 — recorded per item
                out.append(e)
        return out

    def safe(it):
        try:
            return fn(it)
        except Exception as e:          # noqa: BLE001
            return e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(safe, items))
