"""Pipeline tests with fake chains: no network, no API key needed."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pdfsummarizer.chains import Chains, build_chains, message_text
from pdfsummarizer.config import Settings
from pdfsummarizer.summarize import SummarizerError, SummarizerService


def settings(**overrides: object) -> Settings:
    return Settings(api_key="test-key-" + "x" * 32, ocr_enabled=False, **overrides)


def service_with(responses: list[str], **overrides: object) -> SummarizerService:
    config = settings(**overrides)
    llm = FakeListChatModel(responses=responses)
    return SummarizerService(config, chains=build_chains(config, llm=llm))


def flaky_chains() -> Chains:
    def map_chunk(payload: dict) -> str:
        if "BAD" in payload["text"]:
            raise RuntimeError("boom")
        return "chunk summary"

    def fail_reduce(_: dict) -> str:
        raise RuntimeError("reduce down")

    return Chains(
        map=RunnableLambda(map_chunk),
        reduce=RunnableLambda(fail_reduce),
        analysis=RunnableLambda(lambda _: "structure"),
        quotes=RunnableLambda(lambda _: '1. alpha\n- beta\n"gamma"'),
    )


def test_single_chunk_needs_no_reduce() -> None:
    service = service_with(["only summary"])

    result = service.summarize(file_name="a.pdf", text="hello", chunks=["hello"])

    assert result.combined_summary == "only summary"
    assert result.total_chunks == 1
    assert result.successful_chunks == 1
    assert result.failed_chunks == 0


def test_multiple_chunks_are_combined() -> None:
    service = service_with(["first", "second", "MERGED"])

    result = service.summarize(file_name="a.pdf", text="hello", chunks=["one", "two"])

    assert result.combined_summary == "MERGED"
    assert [chunk.summary for chunk in result.chunks] == ["first", "second"]


def test_failed_chunk_is_isolated_and_reported() -> None:
    service = SummarizerService(settings(), chains=flaky_chains())

    result = service.summarize(
        file_name="a.pdf", text="good BAD", chunks=["good", "BAD"]
    )

    assert result.successful_chunks == 1
    assert result.failed_chunks == 1
    assert result.chunks[1].ok is False
    assert "boom" in (result.chunks[1].error or "")
    assert any("Chunk(s) 2 failed" in warning for warning in result.warnings)
    assert result.combined_summary == "chunk summary"


def test_all_chunks_failing_raises() -> None:
    service = SummarizerService(settings(), chains=flaky_chains())

    with pytest.raises(SummarizerError, match="All 1 chunk"):
        service.summarize(file_name="a.pdf", text="BAD", chunks=["BAD"])


def test_analysis_and_quotes_are_optional() -> None:
    plain = service_with(["s"]).summarize(file_name="a.pdf", text="hello", chunks=["hello"])
    enriched = service_with(["s", "structure", '1. alpha\n- beta\n"gamma"']).summarize(
        file_name="a.pdf",
        text="hello",
        chunks=["hello"],
        include_analysis=True,
        include_quotes=True,
    )

    assert plain.analysis is None and plain.quotes == []
    assert enriched.analysis is not None and enriched.analysis.ok
    assert enriched.analysis.text == "structure"
    assert enriched.quotes == ["alpha", "beta", "gamma"]


def test_analysis_failure_is_reported_not_raised() -> None:
    chains = Chains(
        map=RunnableLambda(lambda payload: "s"),
        reduce=RunnableLambda(lambda _: "combined"),
        analysis=RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("nope"))),
        quotes=RunnableLambda(lambda _: ""),
    )

    result = SummarizerService(settings(), chains=chains).summarize(
        file_name="a.pdf", text="hello", chunks=["hello"], include_analysis=True
    )

    assert result.analysis is not None
    assert result.analysis.ok is False
    assert "nope" in result.analysis.text


def test_many_chunks_produce_a_warning() -> None:
    service = service_with(["s"] * 50)

    result = service.summarize(file_name="a.pdf", text="x", chunks=["x"] * 45)

    assert any("chunks is a lot" in warning for warning in result.warnings)


def test_compression_ratio_is_computed() -> None:
    service = service_with(["abcd"])

    result = service.summarize(file_name="a.pdf", text="abcdefgh", chunks=["abcdefgh"])

    assert result.original_chars == 8
    assert result.compression_ratio == 0.5


def test_message_text_handles_strings_and_blocks() -> None:
    assert message_text(AIMessage(content="plain")) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert message_text(blocks) == "ab"
