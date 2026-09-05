"""Regression tests for the SGLang streaming 400 path.

sgl-project/sglang#30917 turned ``return_token_ids`` + ``stream=true`` from an
ignored unknown field into a recognized-but-rejected combination (HTTP 400)
on SGLang's OpenAI-compatible endpoint.

The failure mode that reached the user: the 400 arrives on a *streaming*
request, so ``raise_for_status()`` raises with a response whose body was
never read.  After the ``async with`` block unwinds, the response stream is
closed, and ``await response.aread()`` inside the error handler raises
``httpx.StreamClosed`` ("Attempted to read or stream content, but the stream
has been closed.") — masking the real 400 and skipping the built-in retry
that drops ``return_token_ids``.  The user saw exactly that masked error:
'Error: Attempted to read or stream content, but the stream has been closed.'
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tool_eval_bench.runner.throughput import TokenizerConfig, _stream_one

SGLANG_400_MESSAGE = (
    "return_token_ids is not supported with streaming on /v1/chat/completions. "
    "Please set stream=false when using return_token_ids=true."
)


def _make_sse_line(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


def _sse_ok() -> bytes:
    ok = {"choices": [{"delta": {"content": "ok"}}]}
    return (_make_sse_line(ok) + "data: [DONE]\n\n").encode()


class _UnreadErrorStream(httpx.AsyncByteStream):
    """An error body that is never read before the context manager closes it.

    Mirrors SGLang's behaviour: it answers a streaming request with a plain
    JSON error body, the client raises for status without reading the stream,
    and the ``async with`` teardown closes the response.
    """

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body


@pytest.mark.asyncio
async def test_stream_one_recovers_when_sglang_rejects_stream_token_ids() -> None:
    """SGLang's stream+token_ids 400 must fall back to a clean retry.

    Regression: the error handler's ``aread()`` raised httpx.StreamClosed,
    which propagated out of ``_stream_one`` and aborted the whole spec-bench
    run with the masked 'stream has been closed' error.
    """
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "return_token_ids" in body:
            return httpx.Response(
                400,
                stream=_UnreadErrorStream(
                    json.dumps({"message": SGLANG_400_MESSAGE, "type": "BadRequest"}).encode()
                ),
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            200,
            content=_sse_ok(),
            headers={"content-type": "text/event-stream"},
        )

    cfg = TokenizerConfig()
    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await _stream_one(
            client,
            "http://localhost:8888/v1",
            "qwen3.8-27b-sglang",
            [{"role": "user", "content": "hi"}],
            5,
            None,
            cfg,
        )

    assert sample.error is None
    assert sample.tg_tokens == 1
    assert cfg.supports_return_token_ids is False
    assert "return_token_ids" in bodies[0]
    assert "return_token_ids" not in bodies[1]