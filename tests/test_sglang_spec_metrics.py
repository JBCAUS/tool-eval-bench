"""Regression tests for SGLang speculative-decoding gauges."""

from __future__ import annotations

import json

import httpx
import pytest

from tests.conftest import MeasurementTestClient
from tool_eval_bench.runner.speculative import (
    SpecDecodeInfo,
    detect_spec_decoding,
    measure_spec_single,
    parse_prometheus_spec_metrics,
)

_SGLANG_METRICS = """\
# HELP sglang:spec_accept_length Mean acceptance length of speculative decoding (accepted drafts + bonus token per forward).
# TYPE sglang:spec_accept_length gauge
sglang:spec_accept_length{engine_type="draft",model_name="draft-model",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 2.7
sglang:spec_accept_length{engine_type="unified",model_name="qwen3.8-27b-sglang",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 2.8
# HELP sglang:spec_accept_rate Speculative acceptance rate (`accepted drafts / proposed drafts` in batch).
# TYPE sglang:spec_accept_rate gauge
sglang:spec_accept_rate{engine_type="draft",model_name="draft-model",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.5
sglang:spec_accept_rate{engine_type="unified",model_name="qwen3.8-27b-sglang",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.6
"""


def _sse_response() -> httpx.Response:
    chunks = [
        {
            "choices": [{"delta": {"content": "ok"}, "token_ids": [101]}],
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    body += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.asyncio
async def test_detects_sglang_gauges_and_preserves_method_hint() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=_SGLANG_METRICS))

    async with MeasurementTestClient(transport=transport) as client:
        info = await detect_spec_decoding(
            client,
            "http://host:8888/v1",
            backend_hint="dflash",
        )

    assert info.active is True
    assert info.has_prometheus is True
    assert info.has_sglang_gauges is True
    assert info.method == "dflash"
    assert info.detail == "Detected via Prometheus /metrics (SGLang spec gauges)"


def test_parses_last_sglang_gauge_series_without_summing() -> None:
    metrics = parse_prometheus_spec_metrics(_SGLANG_METRICS)

    assert metrics.sglang_accept_rate == pytest.approx(0.6)
    assert metrics.sglang_accept_length == pytest.approx(2.8)
    assert metrics.accepted_tokens == 0
    assert metrics.draft_tokens == 0
    assert metrics.num_drafts == 0


@pytest.mark.asyncio
async def test_measurement_maps_sglang_gauges_without_counter_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=_SGLANG_METRICS)
        if request.url.path == "/v1/chat/completions":
            return _sse_response()
        return httpx.Response(404)

    info = SpecDecodeInfo(
        active=True,
        method="dflash",
        has_prometheus=True,
        has_sglang_gauges=True,
    )
    async with MeasurementTestClient(transport=httpx.MockTransport(handler)) as client:
        sample = await measure_spec_single(
            client,
            "http://host:8888/v1",
            "qwen3.8-27b-sglang",
            tg=1,
            prompt_type="code",
            spec_info=info,
        )

    assert sample.error is None
    assert sample.acceptance_rate == pytest.approx(0.6)
    assert sample.acceptance_length == pytest.approx(2.8)
    assert sample.draft_tokens_delta is None
    assert sample.accepted_tokens_delta is None
    assert sample.num_drafts_delta is None
    assert sample.draft_tps is None
    assert sample.waste_ratio is None


@pytest.mark.asyncio
async def test_detection_ignores_unrelated_metrics() -> None:
    body = "vllm:prompt_tokens_total 50000\nvllm:generation_tokens_total 12000\n"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))

    async with MeasurementTestClient(transport=transport) as client:
        info = await detect_spec_decoding(client, "http://host:8888/v1")

    assert info.active is False
    assert info.has_prometheus is False
    assert info.has_sglang_gauges is False
