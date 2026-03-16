import re
from collections.abc import Mapping
from typing import Any


REASONING_TAG_PATTERN = re.compile(
    r"<(think|analysis)\b[^>]*>([\s\S]*?)</\1>", re.IGNORECASE
)


def normalize_openai_base_url(base_url: str | None) -> str:
    normalized = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if normalized.endswith("/embeddings"):
        normalized = normalized.rsplit("/", 1)[0]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return f"{normalized}/"


def extract_openai_message_text(response: Any) -> str:
    def get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key, None)

    def normalize_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, Mapping):
                    for field in ("text", "content", "output_text", "reasoning"):
                        candidate = item.get(field)
                        if isinstance(candidate, str) and candidate.strip():
                            parts.append(candidate.strip())
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            return "\n".join(parts).strip()
        return ""

    choices = get_value(response, "choices") or []
    if not choices:
        output_text = normalize_text(get_value(response, "output_text"))
        if output_text:
            return output_text
        return ""

    first_choice = choices[0]
    message = get_value(first_choice, "message")
    if message is None:
        legacy_text = normalize_text(get_value(first_choice, "text"))
        if legacy_text:
            return legacy_text
        return ""

    for field in (
        "content",
        "reasoning_content",
        "reasoning",
        "output_text",
        "text",
    ):
        extracted = normalize_text(get_value(message, field))
        if extracted:
            return extracted

    return ""


def split_reasoning_content(text: str) -> tuple[str, str | None]:
    if not isinstance(text, str) or not text:
        return text, None

    reasoning_parts: list[str] = []

    def _collect_and_remove(match: re.Match[str]) -> str:
        snippet = match.group(2).strip()
        if snippet:
            reasoning_parts.append(snippet)
        return ""

    cleaned = REASONING_TAG_PATTERN.sub(_collect_and_remove, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    thinking = "\n\n".join(reasoning_parts).strip() if reasoning_parts else None
    return cleaned, thinking


def strip_reasoning_content_tags(text: str) -> str:
    cleaned, _ = split_reasoning_content(text)
    return cleaned


def build_openai_naming_request_kwargs(
    temperature: float = 0.7, max_tokens: int = 80
) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        },
    }


def extract_openai_title_text(response: Any) -> str:
    def get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key, None)

    def normalize_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, Mapping):
                    for field in ("text", "content", "output_text"):
                        candidate = item.get(field)
                        if isinstance(candidate, str) and candidate.strip():
                            parts.append(candidate.strip())
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            return "\n".join(parts).strip()
        return ""

    choices = get_value(response, "choices") or []
    if not choices:
        return ""

    first_choice = choices[0]
    message = get_value(first_choice, "message")

    if message is not None:
        for field in ("content", "output_text", "text"):
            title = normalize_text(get_value(message, field))
            if title:
                return title

    return normalize_text(get_value(first_choice, "text"))


def normalize_session_name_candidate(raw_name: str) -> str:
    if not raw_name:
        return ""

    text = strip_reasoning_content_tags(str(raw_name)).strip()

    quote_matches = re.findall(r"[\"“](.{2,40}?)[\"”]", text)
    for quoted in quote_matches:
        value = quoted.strip()
        if value and len(value) <= 20:
            return value

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def clean_line(item: str) -> str:
        value = item.strip("\"'")
        value = re.sub(r"^\*\*(.*?)\*\*$", r"\1", value)
        value = re.sub(r"^`(.*?)`$", r"\1", value)
        value = re.sub(
            r"^(根据.*?(生成|给出).*?(标题|名称)[：:]\s*|(?:最终)?标题[：:]\s*|会话(?:标题|名称)[：:]\s*|title\s*[:：]\s*|这个对话的标题可以.*?[：:]\s*)",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
        return value

    cleaned_candidates: list[str] = []
    for line in lines:
        value = clean_line(line)
        if not value:
            continue
        if re.search(
            r"(字符以内|反映了|核心内容|可以概括为|标题可以|这个标题|这个对话)", value
        ):
            continue
        if value.strip("*#-:： "):
            cleaned_candidates.append(value)

    short_candidates = [c for c in cleaned_candidates if len(c) <= 20]
    if short_candidates:
        return min(short_candidates, key=len)

    if cleaned_candidates:
        return cleaned_candidates[0]

    return ""
