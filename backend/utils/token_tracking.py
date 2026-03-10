import logging

logger = logging.getLogger(__name__)


class _FallbackTokenTracker:
    def add_vanna_tokens(self, **kwargs):
        return None

    def add_tool_usage(self, tool_name, success=True):
        return None


def _fallback_estimate_tokens_from_messages(messages):
    return sum(
        len(msg.get("content", "")) // 4 for msg in messages if isinstance(msg, dict)
    )


def _fallback_create_token_callback(_scope):
    return None


global_token_tracker = _FallbackTokenTracker()
estimate_tokens_from_messages = _fallback_estimate_tokens_from_messages
create_token_callback = _fallback_create_token_callback


try:
    from evaluation.src.token_tracker import (  # type: ignore
        global_token_tracker as eval_token_tracker,
        estimate_tokens_from_messages as eval_estimate_tokens_from_messages,
    )

    global_token_tracker = eval_token_tracker
    estimate_tokens_from_messages = eval_estimate_tokens_from_messages
except Exception:
    logger.info("Evaluation token tracker not found; using lightweight fallback")

try:
    from evaluation.src.langchain_token_callback import (  # type: ignore
        create_token_callback as eval_create_token_callback,
    )

    create_token_callback = eval_create_token_callback
except Exception:
    logger.info("Evaluation langchain callback not found; using no-op callback")
