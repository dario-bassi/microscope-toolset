"""Parse Claude Code JSONL conversation files for dashboard visualization."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    tool_use_id: str
    content: str | list[Any]
    is_error: bool


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    caller: str | None = None


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock


@dataclass
class ParsedUserMessage:
    uuid: str
    timestamp: str
    cwd: str
    git_branch: str
    # Plain text prompt or list of tool results returned to the assistant
    content: str | list[ToolResult]


@dataclass
class ParsedAssistantMessage:
    uuid: str
    timestamp: str
    cwd: str
    git_branch: str
    model: str
    usage: dict[str, Any]
    content_blocks: list[ContentBlock]
    stop_reason: str | None = None


@dataclass
class ParsedFileSnapshot:
    message_id: str
    timestamp: str
    is_snapshot_update: bool


@dataclass
class ParsedQueueOperation:
    timestamp: str
    operation: str
    content: str
    session_id: str


@dataclass
class ParsedLogEntry:
    timestamp: str  # local time as written in the log file
    timestamp_utc: str  # converted to UTC ISO for sorting/comparison
    level: str  # DEBUG / INFO / WARN / ERROR
    source: str  # Core, pymmcore-plus, LogManager, …
    message: str  # may include multi-line traceback


@dataclass
class ParsedMicroscopeLog:
    """Group of hardware log entries that occurred during one tool execution."""

    timestamp: str  # UTC ISO of the first entry (for sorting)
    entries: list[ParsedLogEntry]


@dataclass
class ConversationStats:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    first_timestamp: str = ""
    last_timestamp: str = ""
    models_used: list[str] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    num_tool_calls: int = 0
    num_user_turns: int = 0
    num_assistant_turns: int = 0
    session_id: str = ""
    git_branch: str = ""
    cwd: str = ""


ParsedMessage = (
    ParsedUserMessage
    | ParsedAssistantMessage
    | ParsedFileSnapshot
    | ParsedQueueOperation
    | ParsedMicroscopeLog
)

# ---------------------------------------------------------------------------
# Model pricing  (USD per 1M tokens)
# ---------------------------------------------------------------------------

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
    "claude-haiku-4-5-20251001": {
        "input": 0.8,
        "output": 4.0,
        "cache_read": 0.08,
        "cache_write": 1.0,
    },
    # Legacy / fallback
    "claude-3-5-sonnet-20241022": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-3-opus-20240229": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75}


def _cost_for_usage(model: str, usage: dict[str, Any]) -> float:
    pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    m = 1_000_000
    cost = (
        usage.get("input_tokens", 0) * pricing["input"] / m
        + usage.get("output_tokens", 0) * pricing["output"] / m
        + usage.get("cache_read_input_tokens", 0) * pricing["cache_read"] / m
        + usage.get("cache_creation_input_tokens", 0) * pricing["cache_write"] / m
    )
    return cost


# ---------------------------------------------------------------------------
# Individual message parsers
# ---------------------------------------------------------------------------


def _safe_get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def parse_user_message(raw: dict[str, Any]) -> ParsedUserMessage:
    msg = _safe_get(raw, "message", {})
    raw_content = _safe_get(msg, "content", "")

    if isinstance(raw_content, list):
        content: str | list[ToolResult] = [
            ToolResult(
                tool_use_id=_safe_get(block, "tool_use_id", ""),
                content=_safe_get(block, "content", ""),
                is_error=bool(_safe_get(block, "is_error", False)),
            )
            for block in raw_content
            if isinstance(block, dict)
        ]
    else:
        content = str(raw_content) if raw_content is not None else ""

    return ParsedUserMessage(
        uuid=_safe_get(raw, "uuid", ""),
        timestamp=_safe_get(raw, "timestamp", ""),
        cwd=_safe_get(raw, "cwd", ""),
        git_branch=_safe_get(raw, "gitBranch", ""),
        content=content,
    )


def parse_assistant_message(raw: dict[str, Any]) -> ParsedAssistantMessage:
    msg = _safe_get(raw, "message", {})
    raw_content = _safe_get(msg, "content", [])
    usage = _safe_get(msg, "usage") or {}
    model = _safe_get(msg, "model", "")

    blocks: list[ContentBlock] = []
    for block in raw_content if isinstance(raw_content, list) else []:
        if not isinstance(block, dict):
            continue
        btype = _safe_get(block, "type", "")
        if btype == "text":
            blocks.append(TextBlock(text=_safe_get(block, "text", "")))
        elif btype == "thinking":
            blocks.append(ThinkingBlock(thinking=_safe_get(block, "thinking", "")))
        elif btype == "tool_use":
            blocks.append(
                ToolUseBlock(
                    id=_safe_get(block, "id", ""),
                    name=_safe_get(block, "name", ""),
                    input=_safe_get(block, "input") or {},
                    caller=_safe_get(block, "caller"),
                )
            )
        # Unknown block types are silently skipped

    return ParsedAssistantMessage(
        uuid=_safe_get(raw, "uuid", ""),
        timestamp=_safe_get(raw, "timestamp", ""),
        cwd=_safe_get(raw, "cwd", ""),
        git_branch=_safe_get(raw, "gitBranch", ""),
        model=model,
        usage=usage,
        content_blocks=blocks,
        stop_reason=_safe_get(msg, "stop_reason"),
    )


def parse_file_snapshot(raw: dict[str, Any]) -> ParsedFileSnapshot:
    snapshot = _safe_get(raw, "snapshot", {})
    return ParsedFileSnapshot(
        message_id=_safe_get(raw, "messageId", ""),
        timestamp=_safe_get(snapshot, "timestamp", _safe_get(raw, "timestamp", "")),
        is_snapshot_update=bool(_safe_get(raw, "isSnapshotUpdate", False)),
    )


def parse_queue_operation(raw: dict[str, Any]) -> ParsedQueueOperation:
    return ParsedQueueOperation(
        timestamp=_safe_get(raw, "timestamp", ""),
        operation=_safe_get(raw, "operation", ""),
        content=str(_safe_get(raw, "content", "")),
        session_id=_safe_get(raw, "sessionId", ""),
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PARSERS = {
    "user": parse_user_message,
    "assistant": parse_assistant_message,
    "file-history-snapshot": parse_file_snapshot,
    "queue-operation": parse_queue_operation,
}

_SKIP_TYPES = {"permission-mode", "attachment", "last-prompt", "system"}


def _dispatch(raw: dict[str, Any]) -> ParsedMessage | None:
    msg_type = _safe_get(raw, "type", "")
    if msg_type in _SKIP_TYPES:
        return None
    parser = _PARSERS.get(msg_type)
    if parser is None:
        return None  # Unknown type — skip gracefully
    try:
        return parser(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------


def compute_stats(messages: list[ParsedMessage]) -> ConversationStats:
    stats = ConversationStats()

    timestamps: list[str] = []
    models_seen: set[str] = set()

    for msg in messages:
        ts = getattr(msg, "timestamp", "")
        if ts:
            timestamps.append(ts)

        if isinstance(msg, ParsedUserMessage):
            if isinstance(msg.content, str) and msg.content:
                stats.num_user_turns += 1
            if not stats.git_branch and msg.git_branch:
                stats.git_branch = msg.git_branch
            if not stats.cwd and msg.cwd:
                stats.cwd = msg.cwd

        elif isinstance(msg, ParsedAssistantMessage):
            stats.num_assistant_turns += 1
            if msg.model:
                models_seen.add(msg.model)
            u = msg.usage
            stats.total_input_tokens += u.get("input_tokens", 0)
            stats.total_output_tokens += u.get("output_tokens", 0)
            stats.total_cache_creation_tokens += u.get("cache_creation_input_tokens", 0)
            stats.total_cache_read_tokens += u.get("cache_read_input_tokens", 0)
            stats.estimated_cost_usd += _cost_for_usage(msg.model, u)
            for block in msg.content_blocks:
                if isinstance(block, ToolUseBlock):
                    stats.num_tool_calls += 1

    # Duration: first user text prompt → last assistant message
    first_user_ts = ""
    last_agent_ts = ""
    for msg in messages:
        if (
            isinstance(msg, ParsedUserMessage)
            and isinstance(msg.content, str)
            and msg.content
            and not first_user_ts
            and msg.timestamp
        ):
            first_user_ts = msg.timestamp
        elif isinstance(msg, ParsedAssistantMessage) and msg.timestamp:
            last_agent_ts = msg.timestamp
    stats.first_timestamp = first_user_ts
    stats.last_timestamp = last_agent_ts

    stats.models_used = sorted(models_seen)
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_file(path: str) -> tuple[list[ParsedMessage], ConversationStats]:
    """Read and parse a Claude Code conversation JSONL file.

    Returns a tuple of (parsed_messages, stats). Raises FileNotFoundError
    if the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Conversation file not found: {path!r}")

    raw_rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    messages: list[ParsedMessage] = []
    session_id = ""
    for raw in raw_rows:
        if not session_id:
            session_id = _safe_get(raw, "sessionId", "")
        parsed = _dispatch(raw)
        if parsed is not None:
            messages.append(parsed)

    stats = compute_stats(messages)
    stats.session_id = session_id
    return messages, stats


# ---------------------------------------------------------------------------
# Hardware log parsing & merging
# ---------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+)\s+tid\S+\s+\[(\w+),([^\]]+)\]\s*(.*)")
_LEVEL_MAP = {"IFO": "INFO", "DBG": "DEBUG", "ERR": "ERROR", "WRN": "WARN"}


def _log_ts_to_utc(ts_str: str) -> str:
    """Convert a log-file local timestamp to a UTC ISO string.

    Log files have no timezone suffix — they use the system local time.
    This converts them to UTC so they can be compared to JSONL timestamps.
    """
    try:
        # Normalise to max 6 decimal digits (fromisoformat is strict)
        norm = re.sub(r"(\.\d{1,6})\d*$", r"\1", ts_str.strip())
        naive = datetime.fromisoformat(norm)
        # astimezone() on a naive datetime uses the system's local timezone
        utc_dt = naive.astimezone(UTC)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    except Exception:
        return ts_str


def default_log_path() -> str | None:
    """Return the default pymmcore-plus log path for the current OS."""
    if sys.platform == "win32":
        base = os.environ.get(
            "LOCALAPPDATA", ""
        )  # to change. It depends where micromanager is installed. Make it more flexible.
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Logs")
    else:
        base = os.path.expanduser("~/.local/share")
    path = os.path.join(base, "pymmcore-plus", "pymmcore-plus", "logs", "pymmcore-plus.log")
    return path if os.path.exists(path) else None


def parse_log_file(path: str) -> list[ParsedLogEntry]:
    """Parse a pymmcore-plus log file into a list of ParsedLogEntry objects.

    Handles both 3- and 6-decimal timestamp formats and multi-line tracebacks.
    """
    entries: list[ParsedLogEntry] = []
    current: ParsedLogEntry | None = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            m = _LOG_LINE_RE.match(line)
            if m:
                if current is not None:
                    entries.append(current)
                ts_local = m.group(1)
                level_raw = m.group(2).upper()
                level = _LEVEL_MAP.get(level_raw, level_raw)
                current = ParsedLogEntry(
                    timestamp=ts_local,
                    timestamp_utc=_log_ts_to_utc(ts_local),
                    level=level,
                    source=m.group(3).strip(),
                    message=m.group(4).strip(),
                )
            elif current is not None and line.strip():
                # Continuation line (traceback etc.)
                current.message += "\n" + line

    if current is not None:
        entries.append(current)
    return entries


def merge_logs(
    messages: list[ParsedMessage],
    log_entries: list[ParsedLogEntry],
) -> list[ParsedMessage]:
    """Interleave hardware log blocks into the message list.

    For every user message that carries tool results, collect all log entries
    whose UTC timestamp falls in the window
    [preceding assistant message timestamp, this user message timestamp].
    Those entries are grouped into a ParsedMicroscopeLog and inserted
    immediately after the tool-result user message.
    """
    if not log_entries:
        return messages

    # Sort log entries by UTC timestamp (lexicographic sort works for ISO)
    sorted_logs = sorted(log_entries, key=lambda e: e.timestamp_utc)

    result: list[ParsedMessage] = []
    last_assistant_ts: str = ""

    for msg in messages:
        result.append(msg)

        if isinstance(msg, ParsedAssistantMessage):
            last_assistant_ts = msg.timestamp or ""

        elif (
            isinstance(msg, ParsedUserMessage)
            and isinstance(msg.content, list)
            and last_assistant_ts
            and msg.timestamp
        ):
            window_start = last_assistant_ts
            window_end = msg.timestamp

            matching = [e for e in sorted_logs if window_start <= e.timestamp_utc <= window_end]
            if matching:
                result.append(
                    ParsedMicroscopeLog(
                        timestamp=matching[0].timestamp_utc,
                        entries=matching,
                    )
                )

    return result
