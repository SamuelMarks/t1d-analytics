"""Domain facade coordinating `nl_to_sql_translator`, `chat_turn_executor`, and `sse_chat_streamer`."""

from t1d_analytics.chat_turn_executor import (
    execute_chat_turn as execute_chat_turn,
)
from t1d_analytics.chat_turn_executor import (
    execute_nl_to_sql_eval_turn as execute_nl_to_sql_eval_turn,
)
from t1d_analytics.chat_turn_executor import (
    resolve_db_provenance_flags as resolve_db_provenance_flags,
)
from t1d_analytics.chat_turn_executor import (
    resolve_sql_mode_query as resolve_sql_mode_query,
)
from t1d_analytics.nl_to_sql_translator import (
    TRANSLATION_SCHEMA_CACHE as TRANSLATION_SCHEMA_CACHE,
)
from t1d_analytics.nl_to_sql_translator import (
    build_sql_prompt_with_history as build_sql_prompt_with_history,
)
from t1d_analytics.nl_to_sql_translator import (
    complete_non_streaming_with_fallback as complete_non_streaming_with_fallback,
)
from t1d_analytics.nl_to_sql_translator import (
    extract_completion_or_iterable_text as extract_completion_or_iterable_text,
)
from t1d_analytics.nl_to_sql_translator import (
    extract_schema_for_translation as extract_schema_for_translation,
)
from t1d_analytics.nl_to_sql_translator import (
    invoke_llm_with_local_fallback as invoke_llm_with_local_fallback,
)
from t1d_analytics.nl_to_sql_translator import (
    probe_installed_ollama_names as probe_installed_ollama_names,
)
from t1d_analytics.nl_to_sql_translator import (
    resolve_and_select_local_ollama_model as resolve_and_select_local_ollama_model,
)
from t1d_analytics.nl_to_sql_translator import (
    resolve_effective_retries_for_provider as resolve_effective_retries_for_provider,
)
from t1d_analytics.nl_to_sql_translator import (
    resolve_local_ollama_candidates_list as resolve_local_ollama_candidates_list,
)
from t1d_analytics.nl_to_sql_translator import (
    stream_llm_completion_tokens as stream_llm_completion_tokens,
)
from t1d_analytics.nl_to_sql_translator import (
    translate_nl_to_sql as translate_nl_to_sql,
)
from t1d_analytics.nl_to_sql_translator import (
    validate_or_repair_translated_sql as validate_or_repair_translated_sql,
)
from t1d_analytics.sql_fallback_synthesizer import (
    GenerationModeLiteral as GenerationModeLiteral,
)
from t1d_analytics.sql_fallback_synthesizer import (
    TranslationResult as TranslationResult,
)
from t1d_analytics.sql_fallback_synthesizer import (
    extract_fallback_error_from_text as extract_fallback_error_from_text,
)
from t1d_analytics.sql_fallback_synthesizer import (
    format_llm_failure_reason as format_llm_failure_reason,
)
from t1d_analytics.sql_fallback_synthesizer import (
    is_fallback_response_text as is_fallback_response_text,
)
from t1d_analytics.sql_fallback_synthesizer import (
    synthesize_fallback_clinical_sql as synthesize_fallback_clinical_sql,
)
from t1d_analytics.sse_chat_streamer import (
    DisconnectCheckableProtocol as DisconnectCheckableProtocol,
)
from t1d_analytics.sse_chat_streamer import (
    SqlGoldComparisonPayloadDict as SqlGoldComparisonPayloadDict,
)
from t1d_analytics.sse_chat_streamer import (
    SseDoneEvent as SseDoneEvent,
)
from t1d_analytics.sse_chat_streamer import (
    SseErrorEvent as SseErrorEvent,
)
from t1d_analytics.sse_chat_streamer import (
    SseEventPayload as SseEventPayload,
)
from t1d_analytics.sse_chat_streamer import (
    SseProgressEvent as SseProgressEvent,
)
from t1d_analytics.sse_chat_streamer import (
    SseSqlGeneratedEvent as SseSqlGeneratedEvent,
)
from t1d_analytics.sse_chat_streamer import (
    SseTerminalPayload as SseTerminalPayload,
)
from t1d_analytics.sse_chat_streamer import (
    SseTokenEvent as SseTokenEvent,
)
from t1d_analytics.sse_chat_streamer import (
    finalize_nl_sse_frames as finalize_nl_sse_frames,
)
from t1d_analytics.sse_chat_streamer import (
    format_sse_frame as format_sse_frame,
)
from t1d_analytics.sse_chat_streamer import (
    generate_sse_chat_events as generate_sse_chat_events,
)
from t1d_analytics.sse_chat_streamer import (
    stream_nl_mode_sse_frames as stream_nl_mode_sse_frames,
)

__all__ = [
    "TRANSLATION_SCHEMA_CACHE",
    "DisconnectCheckableProtocol",
    "GenerationModeLiteral",
    "SqlGoldComparisonPayloadDict",
    "SseDoneEvent",
    "SseErrorEvent",
    "SseEventPayload",
    "SseProgressEvent",
    "SseSqlGeneratedEvent",
    "SseTerminalPayload",
    "SseTokenEvent",
    "TranslationResult",
    "build_sql_prompt_with_history",
    "complete_non_streaming_with_fallback",
    "execute_chat_turn",
    "execute_nl_to_sql_eval_turn",
    "extract_completion_or_iterable_text",
    "extract_fallback_error_from_text",
    "extract_schema_for_translation",
    "finalize_nl_sse_frames",
    "format_llm_failure_reason",
    "format_sse_frame",
    "generate_sse_chat_events",
    "invoke_llm_with_local_fallback",
    "is_fallback_response_text",
    "probe_installed_ollama_names",
    "resolve_and_select_local_ollama_model",
    "resolve_db_provenance_flags",
    "resolve_effective_retries_for_provider",
    "resolve_local_ollama_candidates_list",
    "resolve_sql_mode_query",
    "stream_llm_completion_tokens",
    "stream_nl_mode_sse_frames",
    "synthesize_fallback_clinical_sql",
    "translate_nl_to_sql",
    "validate_or_repair_translated_sql",
]
