"""Streaming chunk structural protocols and delta token extraction across LLM providers."""

from collections.abc import Mapping, Sequence
from typing import (
    Protocol,
    TypeGuard,
    runtime_checkable,
)

from typing_extensions import TypedDict


@runtime_checkable
class _TextPartProtocol(Protocol):
    """Protocol for a Google/Gemini content part containing text."""

    @property
    def text(self) -> str | None:
        """Return text fragment."""
        ...


@runtime_checkable
class _CandidateContentProtocol(Protocol):
    """Protocol for a Google/Gemini candidate content block."""

    @property
    def parts(self) -> Sequence[_TextPartProtocol] | None:
        """Return sequence of content parts."""
        ...


@runtime_checkable
class _CandidateProtocol(Protocol):
    """Protocol for a Google/Gemini streaming candidate."""

    @property
    def content(self) -> _CandidateContentProtocol | None:
        """Return candidate content block."""
        ...


@runtime_checkable
class _DeltaTextProtocol(Protocol):
    """Protocol for a streaming delta or choice item exposing `text`."""

    @property
    def text(self) -> str | None:
        """Return delta or choice text."""
        ...


@runtime_checkable
class _DeltaContentProtocol(Protocol):
    """Protocol for a streaming delta or message item exposing `content`."""

    @property
    def content(self) -> str | None:
        """Return delta or message content."""
        ...


_DeltaOrMessageProtocol = _DeltaContentProtocol | _DeltaTextProtocol


@runtime_checkable
class _ChoiceWithDeltaProtocol(Protocol):
    """Protocol for a streaming choice item exposing `delta`."""

    @property
    def delta(
        self,
    ) -> _DeltaContentProtocol | _DeltaTextProtocol | None:
        """Return choice delta."""
        ...


@runtime_checkable
class _ChoiceWithMessageProtocol(Protocol):
    """Protocol for a streaming choice item exposing `message`."""

    @property
    def message(self) -> _DeltaContentProtocol | None:
        """Return choice message."""
        ...


@runtime_checkable
class OpenAiChunkProtocol(Protocol):
    """Protocol for an OpenAI/Ollama/any-llm streaming chunk exposing `choices`."""

    @property
    def choices(
        self,
    ) -> (
        Sequence[
            _ChoiceWithDeltaProtocol | _ChoiceWithMessageProtocol | _DeltaTextProtocol
        ]
        | None
    ):
        """Return sequence of streaming or completion choices."""
        ...


@runtime_checkable
class GoogleChunkProtocol(Protocol):
    """Protocol for a Google/Gemini streaming chunk exposing `candidates`."""

    @property
    def candidates(self) -> Sequence[_CandidateProtocol] | None:
        """Return sequence of streaming candidates."""
        ...


@runtime_checkable
class AnthropicChunkProtocol(Protocol):
    """Protocol for an Anthropic streaming chunk exposing `delta`."""

    @property
    def delta(
        self,
    ) -> _DeltaTextProtocol | _DeltaContentProtocol | None:
        """Return top-level streaming delta."""
        ...


DirectTextChunkProtocol = _DeltaTextProtocol
OpenAiStreamChunkProtocol = OpenAiChunkProtocol
GeminiStreamChunkProtocol = GoogleChunkProtocol
AnthropicStreamChunkProtocol = AnthropicChunkProtocol
LlmStreamChunkProtocol = OpenAiChunkProtocol
LlmDeltaChoicesContainerProtocol = OpenAiChunkProtocol
LlmMessageChoicesContainerProtocol = OpenAiChunkProtocol
GeminiCandidateChunkProtocol = GoogleChunkProtocol
AnthropicDeltaChunkProtocol = AnthropicChunkProtocol

StreamScalar = str | int | float | bool | None


class StreamDeltaMappingDict(TypedDict, total=False):
    """Typed dictionary for a nested `delta` or `message` mapping inside a stream chunk."""

    content: str | None
    text: str | None
    role: str | None


class StreamChoiceMappingDict(TypedDict, total=False):
    """Typed dictionary for a choice item inside a `choices` sequence."""

    delta: StreamDeltaMappingDict
    message: StreamDeltaMappingDict
    text: str | None
    index: int


StreamChoiceElement = (
    StreamScalar
    | StreamChoiceMappingDict
    | Mapping[str, StreamScalar | Mapping[str, StreamScalar]]
)

StreamMappingChunk = Mapping[
    str,
    StreamScalar | Sequence[StreamChoiceElement] | Mapping[str, StreamScalar],
]


@runtime_checkable
class LlmCompletionMessageProtocol(Protocol):
    """Protocol for a message instance inside an LLM completion choice."""

    @property
    def content(self) -> str | None:
        """Return message text content."""
        ...


@runtime_checkable
class LlmCompletionChoiceProtocol(Protocol):
    """Protocol for a completion choice instance exposing a `message`."""

    @property
    def message(self) -> LlmCompletionMessageProtocol:
        """Return completion message."""
        ...


@runtime_checkable
class LlmCompletionResponseProtocol(Protocol):
    """Protocol for a synchronous LLM completion response exposing `choices`."""

    @property
    def choices(self) -> Sequence[LlmCompletionChoiceProtocol]:
        """Return completion choices."""
        ...


ObjectChunkInput = (
    LlmCompletionResponseProtocol
    | OpenAiStreamChunkProtocol
    | GeminiStreamChunkProtocol
    | AnthropicStreamChunkProtocol
    | DirectTextChunkProtocol
)

StreamChunkInput = str | StreamMappingChunk | ObjectChunkInput


def is_mapping_chunk(
    val: StreamChunkInput | StreamChoiceElement | Sequence[StreamChoiceElement],
) -> TypeGuard[StreamMappingChunk]:
    """
    Narrow a streaming chunk or nested element to a string-keyed Mapping.

    Args:
        val: Candidate chunk or nested choice/delta value.

    Returns:
        bool: True if `val` is a Mapping.

    """
    return isinstance(val, Mapping)


def is_sequence_of_choices(
    val: str
    | int
    | float
    | bool
    | Sequence[StreamChoiceElement]
    | Mapping[str, str | int | float | bool | None]
    | None,
) -> TypeGuard[Sequence[StreamChoiceElement]]:
    """
    Narrow a mapping value to a non-string Sequence of choice elements.

    Args:
        val: Raw `choices` field from a mapping chunk.

    Returns:
        bool: True if `val` is a non-string Sequence.

    """
    return isinstance(val, Sequence) and not isinstance(val, str)


def _extract_delta_or_message_str(
    item: _DeltaContentProtocol
    | _DeltaTextProtocol
    | LlmCompletionMessageProtocol
    | None,
    prefer_content: bool = True,
) -> str:
    """
    Extract a non-empty string from a delta or message protocol instance.

    Args:
        item: Delta or message protocol instance.
        prefer_content: Whether to check `.content` before `.text`.

    Returns:
        str: Extracted text or empty string.

    """
    if item is None:
        return ""
    content_val: str | None = (
        item.content if isinstance(item, _DeltaContentProtocol) else None
    )
    text_val: str | None = item.text if isinstance(item, _DeltaTextProtocol) else None
    primary = content_val if prefer_content else text_val
    secondary = text_val if prefer_content else content_val
    chosen = primary or secondary
    return chosen if isinstance(chosen, str) and chosen else ""


def _extract_direct_text(
    item: StreamChunkInput
    | _CandidateProtocol
    | _ChoiceWithDeltaProtocol
    | _ChoiceWithMessageProtocol
    | LlmCompletionChoiceProtocol,
) -> str:
    """
    Extract a non-empty `.text` string from a `DirectTextChunkProtocol` instance.

    Args:
        item: Candidate chunk or choice instance.

    Returns:
        str: Non-empty text string or empty string.

    """
    if isinstance(item, DirectTextChunkProtocol):
        text_val = item.text
        if isinstance(text_val, str) and text_val:
            return text_val
    return ""


def _extract_candidate_text(cand: _CandidateProtocol) -> str:
    """
    Extract text from a single Google/Gemini candidate protocol instance.

    Args:
        cand: Candidate instance from a Gemini chunk.

    Returns:
        str: Extracted candidate text or empty string.

    """
    if not isinstance(cand, _CandidateProtocol) or cand.content is None:
        return ""
    content = cand.content
    if not isinstance(content, _CandidateContentProtocol):
        return ""
    for part in content.parts or ():
        if isinstance(part, _TextPartProtocol):
            text_val = part.text
            if isinstance(text_val, str) and text_val:
                return text_val
    return ""


def _extract_google_chunk_delta(chunk: StreamChunkInput) -> str:
    """
    Extract text delta from a Google/Gemini streaming chunk instance.

    Args:
        chunk: Streaming chunk protocol instance.

    Returns:
        str: Extracted text or empty string.

    """
    if isinstance(chunk, GeminiStreamChunkProtocol):
        for cand in chunk.candidates or ():
            extracted = _extract_candidate_text(cand)
            if extracted:
                return extracted
    return _extract_direct_text(chunk)


def _extract_anthropic_chunk_delta(chunk: StreamChunkInput) -> str:
    """
    Extract text delta from an Anthropic streaming chunk instance.

    Args:
        chunk: Streaming chunk protocol instance.

    Returns:
        str: Extracted text or empty string.

    """
    if isinstance(chunk, AnthropicStreamChunkProtocol) and chunk.delta is not None:
        extracted = _extract_delta_or_message_str(chunk.delta, prefer_content=False)
        if extracted:
            return extracted

    if isinstance(chunk, OpenAiStreamChunkProtocol):
        for c in chunk.choices or ():
            if isinstance(c, _ChoiceWithDeltaProtocol) and c.delta is not None:
                extracted = _extract_delta_or_message_str(c.delta, prefer_content=False)
                if extracted:
                    return extracted
    return ""


def _extract_single_openai_choice_delta(
    choice: _ChoiceWithDeltaProtocol
    | _ChoiceWithMessageProtocol
    | _DeltaTextProtocol
    | LlmCompletionChoiceProtocol,
) -> str:
    """
    Extract text from a single OpenAI/Ollama choice item.

    Args:
        choice: Single choice item from `chunk.choices`.

    Returns:
        str: Extracted choice text or empty string.

    """
    if isinstance(choice, _ChoiceWithDeltaProtocol) and choice.delta is not None:
        extracted = _extract_delta_or_message_str(choice.delta, prefer_content=True)
        if extracted:
            return extracted

    if isinstance(choice, _ChoiceWithMessageProtocol) and choice.message is not None:
        extracted = _extract_delta_or_message_str(choice.message, prefer_content=True)
        if extracted:
            return extracted

    return _extract_direct_text(choice)


def _extract_openai_chunk_delta(chunk: StreamChunkInput) -> str:
    """
    Extract text delta from an OpenAI/Ollama/any-llm streaming chunk instance.

    Args:
        chunk: Streaming chunk protocol instance.

    Returns:
        str: Extracted text or empty string.

    """
    choices = (
        chunk.choices
        if isinstance(chunk, OpenAiStreamChunkProtocol) and chunk.choices is not None
        else ()
    )
    for choice in choices:
        extracted = _extract_single_openai_choice_delta(choice)
        if extracted:
            return extracted

    return _extract_direct_text(chunk)


def _extract_mapping_chunk_delta(chunk: StreamMappingChunk) -> str:
    """
    Extract token delta string from a dictionary/mapping streaming chunk.

    Args:
        chunk: Mapping chunk payload.

    Returns:
        str: Extracted token string or empty string.

    """
    choices_raw = chunk.get("choices")
    if is_sequence_of_choices(choices_raw):
        for choice in choices_raw:
            if is_mapping_chunk(choice):
                delta = choice.get("delta")
                if is_mapping_chunk(delta) and delta.get("content"):
                    return str(delta["content"])
                if choice.get("text"):
                    return str(choice["text"])
    msg = chunk.get("message")
    if is_mapping_chunk(msg) and msg.get("content"):
        return str(msg["content"])
    if chunk.get("response"):
        return str(chunk["response"])
    if chunk.get("text"):
        return str(chunk["text"])
    return ""


def extract_stream_delta(chunk: StreamChunkInput, provider: str = "ollama") -> str:
    """
    Extract token delta string from provider-specific streaming response chunk.

    Accurately inspects delta instances across ollama, openai, anthropic, google, and other any-llm providers.

    Args:
        chunk: Raw response chunk from any-llm or provider SDK.
        provider: Provider identifier ('ollama', 'openai', 'anthropic', 'google', etc.).

    Returns:
        str: Extracted token content string, or empty string.

    """
    if isinstance(chunk, str):
        return chunk
    if is_mapping_chunk(chunk):
        return _extract_mapping_chunk_delta(chunk)

    prov = provider.lower()
    if prov in ("google", "gemini"):
        return _extract_google_chunk_delta(chunk)
    if prov == "anthropic":
        return _extract_anthropic_chunk_delta(chunk)
    return _extract_openai_chunk_delta(chunk)


__all__ = [
    "AnthropicChunkProtocol",
    "AnthropicDeltaChunkProtocol",
    "AnthropicStreamChunkProtocol",
    "DirectTextChunkProtocol",
    "GeminiCandidateChunkProtocol",
    "GeminiStreamChunkProtocol",
    "GoogleChunkProtocol",
    "LlmCompletionChoiceProtocol",
    "LlmCompletionMessageProtocol",
    "LlmCompletionResponseProtocol",
    "LlmDeltaChoicesContainerProtocol",
    "LlmMessageChoicesContainerProtocol",
    "LlmStreamChunkProtocol",
    "ObjectChunkInput",
    "OpenAiChunkProtocol",
    "OpenAiStreamChunkProtocol",
    "StreamChoiceElement",
    "StreamChoiceMappingDict",
    "StreamChunkInput",
    "StreamDeltaMappingDict",
    "StreamMappingChunk",
    "StreamScalar",
    "_CandidateContentProtocol",
    "_CandidateProtocol",
    "_ChoiceWithDeltaProtocol",
    "_ChoiceWithMessageProtocol",
    "_DeltaContentProtocol",
    "_DeltaOrMessageProtocol",
    "_DeltaTextProtocol",
    "_TextPartProtocol",
    "_extract_anthropic_chunk_delta",
    "_extract_candidate_text",
    "_extract_delta_or_message_str",
    "_extract_direct_text",
    "_extract_google_chunk_delta",
    "_extract_mapping_chunk_delta",
    "_extract_openai_chunk_delta",
    "_extract_single_openai_choice_delta",
    "extract_stream_delta",
    "is_mapping_chunk",
    "is_sequence_of_choices",
]
