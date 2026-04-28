"""PyQt6 dashboard for reviewing Claude Code conversation JSONL files."""

from __future__ import annotations

import base64
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import markdown as _md_lib
    _HAS_MARKDOWN = True
except ImportError:
    _HAS_MARKDOWN = False

from PyQt6.QtCore import Qt, QByteArray, QSize, QSizeF
from PyQt6.QtGui import QFont, QPixmap, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

_BENCHMARKING_DIR = Path(__file__).parent

from src.benchmarking.review_conversation import (
    ConversationStats,
    ParsedAssistantMessage,
    ParsedFileSnapshot,
    ParsedLogEntry,
    ParsedMessage,
    ParsedMicroscopeLog,
    ParsedQueueOperation,
    ParsedUserMessage,
    TextBlock,
    ThinkingBlock,
    ToolResult,
    ToolUseBlock,
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

USER_BG      = "#DBEAFE"
ASSISTANT_BG = "#F3F4F6"
SYSTEM_BG    = "#FEF9C3"
THINKING_BG  = "#EDE9FE"
TOOL_USE_BG  = "#D1FAE5"
TOOL_OK_BG   = "#ECFDF5"
TOOL_ERR_BG  = "#FEE2E2"
LOG_BG       = "#E0F2FE"   # light sky-blue for hardware logs
CODE_BG      = "#1E1E1E"
CODE_FG      = "#D4D4D4"
WINDOW_BG    = "#FFFFFF"

_LOG_LEVEL_STYLE: dict[str, tuple[str, str]] = {
    # level → (badge-bg, badge-text)
    "DEBUG": ("#E5E7EB", "#374151"),
    "INFO":  ("#DBEAFE", "#1E40AF"),
    "WARN":  ("#FEF3C7", "#92400E"),
    "ERROR": ("#FEE2E2", "#991B1B"),
}

MAX_COLLAPSED_CHARS = 500

# Approximate content width used to pre-compute document heights
_CONTENT_WIDTH = 820

# ---------------------------------------------------------------------------
# Number / cost formatting  (European style)
# ---------------------------------------------------------------------------

def _fmt_num(n: int) -> str:
    s = str(abs(n))
    groups: list[str] = []
    while len(s) > 3:
        groups.append(s[-3:])
        s = s[:-3]
    groups.append(s)
    return "'".join(reversed(groups))


def _fmt_cost(c: float) -> str:
    return f"{c:.2f}".replace(".", ",")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%Y-%m-%d  %H:%M:%S")
    except Exception:
        return ts


def _short_cwd(cwd: str) -> str:
    """Return only the immediate project folder name."""
    if not cwd:
        return ""
    parts = [p for p in cwd.replace("\\", "/").split("/") if p]
    return parts[-1] if parts else cwd


def _duration(first: str, last: str) -> str:
    if not first or not last:
        return "—"
    try:
        t0 = datetime.fromisoformat(first.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(last.replace("Z", "+00:00"))
        secs = int((t1 - t0).total_seconds())
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"
    except Exception:
        return "—"


def _md_to_html(text: str) -> str:
    if _HAS_MARKDOWN:
        return _md_lib.markdown(
            text,
            extensions=["fenced_code", "tables", "nl2br", "sane_lists"],
        )
    import html as _html
    t = _html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`",       r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


def _doc_height(widget: QTextBrowser | QTextEdit, width: int = _CONTENT_WIDTH) -> int:
    """Compute document pixel height at a given render width."""
    widget.document().setPageSize(QSizeF(width, 10_000))
    return int(widget.document().size().height())


# ---------------------------------------------------------------------------
# Low-level content widgets  (no internal scrollbars — full content height)
# ---------------------------------------------------------------------------

def _make_markdown_widget(text: str, bg: str) -> QTextBrowser:
    w = QTextBrowser()
    w.setReadOnly(True)
    w.setOpenExternalLinks(False)
    w.setFrameStyle(0)
    w.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # setDefaultStyleSheet has better support than inline <style> in Qt's renderer
    w.document().setDefaultStyleSheet(f"""
        body  {{ background-color:{bg}; font-size:12px;
                 font-family:'Segoe UI',Arial,sans-serif; color:#1F2937; }}
        p     {{ margin:2px 0 4px 0; }}
        b, strong {{ color:#111827; }}
        code  {{ background-color:#E8EAF0; padding:1px 4px; border-radius:3px;
                 font-family:Consolas,monospace; font-size:11px; color:#C0392B; }}
        pre   {{ background-color:{CODE_BG}; color:{CODE_FG}; padding:8px;
                 border-radius:6px; margin:4px 0; font-family:Consolas,monospace;
                 font-size:11px; display:block; }}
        pre code {{ background-color:{CODE_BG}; color:{CODE_FG}; padding:0;
                    border-radius:0; }}
        h1,h2 {{ font-size:14px; margin:6px 0 2px 0; }}
        h3    {{ font-size:13px; margin:4px 0 2px 0; }}
        ul,ol {{ margin:2px 0; padding-left:20px; }}
        li    {{ margin:1px 0; }}
        blockquote {{ border-left:3px solid #9CA3AF; padding-left:8px;
                      color:#6B7280; margin:4px 0; }}
        table {{ border-collapse:collapse; margin:4px 0; }}
        th,td {{ border:1px solid #D1D5DB; padding:4px 8px; }}
        th    {{ background-color:rgba(0,0,0,0.05); }}
    """)
    w.setHtml(f"<body>{_md_to_html(text)}</body>")
    w.setStyleSheet(f"background-color:{bg}; border:none;")
    h = _doc_height(w) + 12
    w.setFixedHeight(h)
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return w


def _make_plain_text_widget(text: str, bg: str) -> QTextEdit:
    """Monospace, light background, full-height (no internal scroll)."""
    w = QTextEdit()
    w.setReadOnly(True)
    w.setFrameStyle(0)
    w.setPlainText(text)
    w.setFont(QFont("Consolas", 9))
    w.setStyleSheet(f"background-color:{bg}; color:#1F2937; border:none; padding:2px;")
    w.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
    w.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    h = _doc_height(w) + 12
    w.setFixedHeight(h)
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return w


def _make_code_widget(code: str) -> QTextEdit:
    """Dark monospace widget for explicit fenced code blocks."""
    w = QTextEdit()
    w.setReadOnly(True)
    w.setFrameStyle(0)
    w.setPlainText(code)
    w.setFont(QFont("Consolas", 9))
    w.setStyleSheet(
        f"background-color:{CODE_BG}; color:{CODE_FG}; "
        "border-radius:6px; padding:6px; border:none;"
    )
    w.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
    w.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    h = _doc_height(w) + 12
    w.setFixedHeight(h)
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return w


# ---------------------------------------------------------------------------
# Tool parameter table
# ---------------------------------------------------------------------------

def _make_param_table(params: dict[str, Any], bg: str) -> QWidget:
    """Render tool input as a labelled key/value table."""
    container = QWidget()
    container.setStyleSheet(f"background-color:{bg};")
    grid = QVBoxLayout(container)
    grid.setContentsMargins(0, 4, 0, 0)
    grid.setSpacing(3)

    for key, value in params.items():
        row = QHBoxLayout()
        row.setSpacing(8)

        key_lbl = QLabel(key)
        key_lbl.setFixedWidth(110)
        key_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        key_lbl.setStyleSheet(
            "font-size:10px; font-weight:bold; color:#065F46; "
            f"background:{bg}; padding-top:2px;"
        )
        row.addWidget(key_lbl)

        val_str = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

        # Multi-line or long values get a text widget; short ones get a label
        if "\n" in val_str or len(val_str) > 120:
            val_w = _make_plain_text_widget(val_str, bg)
        else:
            val_w = QLabel(val_str)
            val_w.setWordWrap(True)
            val_w.setStyleSheet(f"font-size:11px; color:#1F2937; background:{bg};")
            val_w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        row.addWidget(val_w, stretch=1)
        grid.addLayout(row)

        # Thin separator between rows
        if key != list(params.keys())[-1]:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"color:rgba(0,0,0,0.08); background:transparent;")
            grid.addWidget(sep)

    return container


# ---------------------------------------------------------------------------
# Collapsible content wrapper
# ---------------------------------------------------------------------------

class CollapsibleWidget(QWidget):
    def __init__(self, text: str, bg: str, is_code: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._text = text
        self._bg = bg
        self._is_code = is_code
        self._expanded = False
        self._needs_collapse = len(text) > MAX_COLLAPSED_CHARS
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._build(expanded=False)

    def _build(self, expanded: bool) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        display = (
            self._text if (expanded or not self._needs_collapse)
            else self._text[:MAX_COLLAPSED_CHARS] + "…"
        )
        w = _make_code_widget(display) if self._is_code else _make_markdown_widget(display, self._bg)
        self._layout.addWidget(w)
        if self._needs_collapse:
            btn = QPushButton("View all" if not expanded else "Close")
            btn.setFixedWidth(80)
            btn.setStyleSheet(
                "QPushButton{font-size:10px;color:#4B5563;background:transparent;"
                "border:1px solid #D1D5DB;border-radius:3px;padding:1px 4px;}"
                "QPushButton:hover{background:#E5E7EB;}"
            )
            btn.clicked.connect(self._toggle)
            self._layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._build(expanded=self._expanded)


# ---------------------------------------------------------------------------
# Block renderers
# ---------------------------------------------------------------------------

def _render_text_block(text: str, bg: str, layout: QVBoxLayout) -> None:
    layout.addWidget(CollapsibleWidget(text, bg=bg))


def _render_thinking_block(thinking: str, layout: QVBoxLayout) -> None:
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame{{background-color:{THINKING_BG};border-radius:8px;}}"
        " QLabel{background-color:transparent;}"
    )
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(8, 6, 8, 6)
    if thinking:
        fl.addWidget(QLabel("<i>💭 Thinking</i>",
                            styleSheet="color:#6D28D9;font-size:10px;"))
        fl.addWidget(CollapsibleWidget(thinking, bg=THINKING_BG))
    else:
        fl.addWidget(QLabel("<i>💭 Thinking (content not stored)</i>",
                            styleSheet="color:#9CA3AF;font-size:10px;"))
    layout.addWidget(frame)


def _render_tool_use_block(name: str, inp: dict[str, Any], layout: QVBoxLayout) -> None:
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame{{background-color:{TOOL_USE_BG};border-radius:8px;}}"
        " QLabel{background-color:transparent;}"
    )
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(10, 8, 10, 10)
    fl.setSpacing(6)

    # Header
    hdr = QHBoxLayout()
    hdr.addWidget(QLabel("🔧", styleSheet="font-size:13px;"))
    hdr.addWidget(QLabel(f"<b>{name}</b>",
                         styleSheet="font-size:12px;color:#065F46;"))
    hdr.addStretch()
    fl.addLayout(hdr)

    # Separator
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("color:rgba(0,100,0,0.15);")
    fl.addWidget(sep)

    # Key/value parameter table
    fl.addWidget(_make_param_table(inp, TOOL_USE_BG))
    layout.addWidget(frame)


def _make_image_widget(b64_data: str) -> QLabel:
    """Render a base64-encoded PNG/JPEG image as a scaled QLabel."""
    raw = base64.b64decode(b64_data)
    ba = QByteArray(raw)
    pixmap = QPixmap()
    pixmap.loadFromData(ba)
    if not pixmap.isNull() and pixmap.width() > _CONTENT_WIDTH - 20:
        pixmap = pixmap.scaledToWidth(
            _CONTENT_WIDTH - 20, Qt.TransformationMode.SmoothTransformation
        )
    label = QLabel()
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    return label


_PERSISTED_RE = re.compile(
    r"<persisted-output>\s*Output too large \(([^)]+)\)[^\n]*\n"
    r".*?Preview \([^)]+\):\n(.*)",
    re.DOTALL,
)


def _clean_persisted_output(text: str) -> tuple[str, bool]:
    """Strip <persisted-output> wrapper and return (cleaned_text, was_truncated).

    If the text is a persisted-output placeholder, returns the preview with a
    short header noting the truncation. Otherwise returns the original text.
    """
    stripped = text.strip()
    if not stripped.startswith("<persisted-output>"):
        return text, False
    m = _PERSISTED_RE.search(stripped)
    if m:
        size, preview = m.group(1), m.group(2).rstrip()
        return f"[Output truncated — full size {size}]\n\n{preview}", True
    # Malformed tag — just strip the XML markers
    cleaned = re.sub(r"</?persisted-output>", "", stripped).strip()
    return cleaned, True


def _render_tool_result(result: ToolResult, layout: QVBoxLayout) -> None:
    color    = TOOL_ERR_BG if result.is_error else TOOL_OK_BG
    txt_col  = "#991B1B"   if result.is_error else "#065F46"
    icon     = "❌" if result.is_error else "✅"
    label    = "Tool error" if result.is_error else "Tool result"

    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame{{background-color:{color};border-radius:8px;}}"
        " QLabel{background-color:transparent;}"
    )
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(10, 8, 10, 10)
    fl.setSpacing(6)

    # Header
    hdr = QHBoxLayout()
    hdr.addWidget(QLabel(icon,  styleSheet="font-size:13px;"))
    hdr.addWidget(QLabel(f"<b>{label}</b>",
                         styleSheet=f"font-size:12px;color:{txt_col};"))
    hdr.addWidget(QLabel(
        f"<span style='color:#9CA3AF;font-size:10px'>{result.tool_use_id[:14]}…</span>"
    ))
    hdr.addStretch()
    fl.addLayout(hdr)

    # Normalise content into renderable segments.
    # Each segment is one of:
    #   ('text',  str,  bool)   — text string, True if should render as markdown
    #   ('image', str)          — base64 image data
    segments: list[tuple] = []

    if isinstance(result.content, list):
        text_parts: list[str] = []
        for item in result.content:
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "")
            if itype == "text":
                text_parts.append(item.get("text", ""))
            elif itype == "image":
                # Flush accumulated text first
                if text_parts:
                    joined = "\n\n".join(text_parts).strip()
                    segments.append(("text", joined, True))   # markdown
                    text_parts = []
                segments.append(("image", item.get("data", "")))
        if text_parts:
            joined = "\n\n".join(text_parts).strip()
            segments.append(("text", joined, True))
    else:
        # Plain string — may be a <persisted-output> placeholder
        cleaned, was_truncated = _clean_persisted_output(result.content)
        segments.append(("text", cleaned, False))   # plain text

    if not segments or all(
        (s[0] == "text" and not s[1].strip()) for s in segments
    ):
        fl.addWidget(QLabel("<i>(empty response)</i>"))
    else:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:rgba(0,0,0,0.08);")
        fl.addWidget(sep)
        for seg in segments:
            if seg[0] == "image":
                try:
                    fl.addWidget(_make_image_widget(seg[1]))
                except Exception:
                    fl.addWidget(QLabel("<i>[image — could not decode]</i>"))
            else:
                _, text, as_markdown = seg
                if not text.strip():
                    continue
                if as_markdown:
                    fl.addWidget(CollapsibleWidget(text, bg=color))
                else:
                    fl.addWidget(_make_plain_text_widget(text, color))

    layout.addWidget(frame)


# ---------------------------------------------------------------------------
# Microscope log bubble
# ---------------------------------------------------------------------------

def _short_log_ts(ts: str) -> str:
    """Show only HH:MM:SS.mmm from a log timestamp."""
    try:
        t = ts.split("T")[1] if "T" in ts else ts
        return t[:12]   # HH:MM:SS.mmm
    except Exception:
        return ts


class LogBubble(QFrame):
    """System-side bubble showing hardware log entries for one tool execution."""

    def __init__(self, msg: ParsedMicroscopeLog, parent: QWidget | None = None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            LogBubble {{
                background-color: {LOG_BG};
                border-radius: 14px;
            }}
            LogBubble QLabel {{
                background-color: transparent;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("🔬  <b>Microscope Log</b>",
                             styleSheet="font-size:11px;color:#0369A1;"))
        hdr.addWidget(QLabel(f"{len(msg.entries)} entries",
                             styleSheet="font-size:10px;color:#6B7280;"))
        hdr.addStretch()
        hdr.addWidget(QLabel(_fmt_ts(msg.timestamp),
                             styleSheet="font-size:10px;color:#6B7280;"))
        outer.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color:#BAE6FD;border:none;max-height:1px;")
        outer.addWidget(sep)

        # Entry rows
        for entry in msg.entries:
            self._add_entry(outer, entry)

    def _add_entry(self, layout: QVBoxLayout, entry: ParsedLogEntry) -> None:
        row_w = QWidget()
        row_w.setStyleSheet(f"background-color:{LOG_BG};")
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(6)

        # Timestamp (short)
        ts_lbl = QLabel(_short_log_ts(entry.timestamp))
        ts_lbl.setFixedWidth(90)
        ts_lbl.setStyleSheet("font-size:10px;color:#6B7280;font-family:Consolas;")
        row.addWidget(ts_lbl)

        # Level badge
        badge_bg, badge_fg = _LOG_LEVEL_STYLE.get(entry.level, ("#E5E7EB", "#374151"))
        level_lbl = QLabel(entry.level)
        level_lbl.setFixedWidth(44)
        level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        level_lbl.setStyleSheet(
            f"font-size:9px;font-weight:bold;color:{badge_fg};"
            f"background-color:{badge_bg};border-radius:3px;padding:1px 2px;"
        )
        row.addWidget(level_lbl)

        # Source
        src_lbl = QLabel(entry.source)
        src_lbl.setFixedWidth(90)
        src_lbl.setStyleSheet("font-size:10px;color:#0369A1;")
        row.addWidget(src_lbl)

        # Message (collapsible only for multi-line tracebacks)
        if "\n" in entry.message:
            msg_w = CollapsibleWidget(entry.message, bg=LOG_BG)
        else:
            msg_w = QLabel(entry.message)
            msg_w.setWordWrap(True)
            msg_w.setStyleSheet("font-size:11px;color:#1F2937;background:transparent;")
            msg_w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        row.addWidget(msg_w, stretch=1)
        layout.addWidget(row_w)


# ---------------------------------------------------------------------------
# Message bubble
# ---------------------------------------------------------------------------

class MessageBubble(QFrame):
    def __init__(self, msg: ParsedMessage, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup(msg)

    def _setup(self, msg: ParsedMessage) -> None:
        if isinstance(msg, ParsedUserMessage):
            bg, role = USER_BG, "User"
        elif isinstance(msg, ParsedAssistantMessage):
            bg, role = ASSISTANT_BG, "Agent"
        else:
            bg, role = SYSTEM_BG, "System"

        self.setStyleSheet(f"""
            MessageBubble {{
                background-color:{bg};
                border-radius:14px;
            }}
            MessageBubble QLabel {{
                background-color:transparent;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"<b>{role}</b>",
                             styleSheet="font-size:11px;color:#111827;"))
        ts = getattr(msg, "timestamp", "")
        if ts:
            hdr.addWidget(QLabel(_fmt_ts(ts),
                                 styleSheet="font-size:10px;color:#6B7280;"))
        hdr.addStretch()
        if isinstance(msg, ParsedAssistantMessage) and msg.usage:
            u = msg.usage
            hdr.addWidget(QLabel(
                f"↓{_fmt_num(u.get('input_tokens',0))}  "
                f"↑{_fmt_num(u.get('output_tokens',0))} tok",
                styleSheet="font-size:10px;color:#6B7280;",
            ))
        outer.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color:#E5E7EB;border:none;max-height:1px;")
        outer.addWidget(sep)

        # Content
        if isinstance(msg, ParsedUserMessage):
            if isinstance(msg.content, str):
                _render_text_block(msg.content, bg, outer)
            else:
                for result in msg.content:
                    _render_tool_result(result, outer)
        elif isinstance(msg, ParsedAssistantMessage):
            for block in msg.content_blocks:
                if isinstance(block, TextBlock):
                    _render_text_block(block.text, bg, outer)
                elif isinstance(block, ThinkingBlock):
                    _render_thinking_block(block.thinking, outer)
                elif isinstance(block, ToolUseBlock):
                    _render_tool_use_block(block.name, block.input, outer)
        elif isinstance(msg, ParsedFileSnapshot):
            outer.addWidget(QLabel(
                f"<i>File snapshot {'updated' if msg.is_snapshot_update else 'saved'}</i>"
            ))
        elif isinstance(msg, ParsedQueueOperation):
            outer.addWidget(QLabel(f"<i>Queue: {msg.operation}</i>"))
            if msg.content:
                outer.addWidget(CollapsibleWidget(msg.content, bg=bg))


# ---------------------------------------------------------------------------
# Stats panel
# ---------------------------------------------------------------------------

class StatsPanel(QWidget):
    def __init__(self, stats: ConversationStats, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(230)
        self.setMaximumWidth(340)
        self.setStyleSheet("background-color:#F9FAFB;border-left:1px solid #E5E7EB;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 18, 14, 18)
        layout.setSpacing(8)

        layout.addWidget(QLabel("<b>Experiment Statistics</b>",
                                styleSheet="font-size:13px;"))
        layout.addWidget(self._sep())

        sid = stats.session_id
        self._row(layout, "Session",     sid[:16] + "…" if len(sid) > 16 else sid)
        self._row(layout, "Git branch",  stats.git_branch or "—")
        if stats.cwd:
            self._row(layout, "Working dir", _short_cwd(stats.cwd))
        self._row(layout, "Duration",
                  _duration(stats.first_timestamp, stats.last_timestamp))

        layout.addWidget(self._sep())
        layout.addWidget(QLabel("<b>Tokens</b>"))
        self._row(layout, "Input",       _fmt_num(stats.total_input_tokens))
        self._row(layout, "Output",      _fmt_num(stats.total_output_tokens))
        self._row(layout, "Cache write", _fmt_num(stats.total_cache_creation_tokens))
        self._row(layout, "Cache read",  _fmt_num(stats.total_cache_read_tokens))

        layout.addWidget(self._sep())
        layout.addWidget(QLabel("<b>Cost (est.)</b>"))
        self._row(layout, "Total USD",   _fmt_cost(stats.estimated_cost_usd))

        layout.addWidget(self._sep())
        layout.addWidget(QLabel("<b>Activity</b>"))
        self._row(layout, "User turns",  _fmt_num(stats.num_user_turns))
        self._row(layout, "Agent turns", _fmt_num(stats.num_assistant_turns))
        self._row(layout, "Tool calls",  _fmt_num(stats.num_tool_calls))

        layout.addWidget(self._sep())
        layout.addWidget(QLabel("<b>Models</b>"))
        for m in (stats.models_used or ["—"]):
            lbl = QLabel(f"• {m}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size:11px;")
            layout.addWidget(lbl)

        layout.addStretch()

    @staticmethod
    def _sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color:#E5E7EB;")
        return f

    @staticmethod
    def _row(layout: QVBoxLayout, label: str, value: str) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(label, styleSheet="font-size:11px;color:#6B7280;"))
        row.addStretch()
        val = QLabel(value)
        val.setStyleSheet("font-size:11px;color:#111827;")
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(val)
        layout.addLayout(row)


# ---------------------------------------------------------------------------
# Grading window
# ---------------------------------------------------------------------------

class GradingWindow(QMainWindow):
    """Popup window for scoring a test run against grading.json criteria."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Test Grading")
        self.resize(680, 780)

        self._spinboxes:  dict[str, tuple[QSpinBox, int]] = {}
        self._checkboxes: dict[str, QCheckBox] = {}
        self._current_test: Path | None = None
        self._criteria: list[dict] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ── Test selector ────────────────────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("<b>Test:</b>"))
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_test_selected)
        sel_row.addWidget(self._combo, 1)
        root.addLayout(sel_row)

        # ── Agent prompt (read-only) ─────────────────────────────────────────
        root.addWidget(QLabel("<b>Agent Prompt:</b>"))
        self._prompt_display = QTextEdit()
        self._prompt_display.setReadOnly(True)
        self._prompt_display.setFixedHeight(110)
        self._prompt_display.setStyleSheet(
            "background:#F3F4F6; border:1px solid #D1D5DB; border-radius:4px; font-size:11px;"
        )
        root.addWidget(self._prompt_display)

        # ── Criteria scroll area ─────────────────────────────────────────────
        root.addWidget(QLabel("<b>Criteria:</b>"))
        self._criteria_widget = QWidget()
        self._criteria_layout = QVBoxLayout(self._criteria_widget)
        self._criteria_layout.setContentsMargins(0, 0, 0, 0)
        self._criteria_layout.setSpacing(4)

        crit_scroll = QScrollArea()
        crit_scroll.setWidgetResizable(True)
        crit_scroll.setWidget(self._criteria_widget)
        crit_scroll.setStyleSheet("border:1px solid #E5E7EB; border-radius:4px;")
        root.addWidget(crit_scroll, 1)

        # ── Score summary ────────────────────────────────────────────────────
        self._score_label = QLabel("Score: 0 / 0  (0%)")
        self._score_label.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#111827;"
        )
        root.addWidget(self._score_label)

        # ── Notes ────────────────────────────────────────────────────────────
        root.addWidget(QLabel("<b>Notes:</b>"))
        self._notes = QTextEdit()
        self._notes.setFixedHeight(72)
        self._notes.setPlaceholderText("Optional notes about this grading…")
        self._notes.setStyleSheet(
            "border:1px solid #D1D5DB; border-radius:4px; font-size:11px;"
        )
        root.addWidget(self._notes)

        # ── Save button ──────────────────────────────────────────────────────
        save_btn = QPushButton("Save Grade")
        save_btn.setStyleSheet(
            "QPushButton{background:#2563EB;color:white;font-weight:bold;"
            "border-radius:6px;padding:6px 18px;font-size:12px;}"
            "QPushButton:hover{background:#1D4ED8;}"
        )
        save_btn.clicked.connect(self._save_grade)
        root.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self._populate_tests()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _populate_tests(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        tests = sorted(
            (p.parent for p in _BENCHMARKING_DIR.glob("test_*/grading.json")),
            key=lambda p: int(p.name.split("_")[1]),
        )
        for test_dir in tests:
            self._combo.addItem(test_dir.name, userData=test_dir)
        self._combo.blockSignals(False)
        if self._combo.count():
            self._on_test_selected(0)

    def _on_test_selected(self, index: int) -> None:
        test_dir: Path | None = self._combo.itemData(index)
        if test_dir is None:
            return
        grading_path = test_dir / "grading.json"
        try:
            data: dict = json.loads(grading_path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Could not read grading.json:\n{exc}")
            return

        self._current_test = test_dir
        self._criteria = data.get("criteria", [])
        self._prompt_display.setPlainText(data.get("agent_prompt", ""))

        # Rebuild criteria widgets
        while self._criteria_layout.count():
            item = self._criteria_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._spinboxes.clear()
        self._checkboxes.clear()

        for crit in self._criteria:
            cid   = crit["id"]
            name  = crit["name"]
            ctype = crit.get("type", "scored")

            row_w = QWidget()
            row_w.setStyleSheet(
                "QWidget{background:#F9FAFB; border-radius:4px;}"
                " QLabel{background:transparent;}"
            )
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(10, 6, 10, 6)
            rl.setSpacing(10)

            name_lbl = QLabel(name)
            name_lbl.setWordWrap(True)
            name_lbl.setStyleSheet("font-size:11px;")
            rl.addWidget(name_lbl, 1)

            if ctype == "pass_fail":
                cb = QCheckBox("Pass")
                cb.setStyleSheet("font-size:11px;")
                cb.toggled.connect(self._update_score)
                rl.addWidget(cb)
                self._checkboxes[cid] = cb
            else:
                max_pts = int(crit.get("max_points", 10))
                sb = QSpinBox()
                sb.setMinimum(0)
                sb.setMaximum(max_pts)
                sb.setFixedWidth(70)
                sb.setSuffix(f" / {max_pts}")
                sb.valueChanged.connect(self._update_score)
                rl.addWidget(sb)
                self._spinboxes[cid] = (sb, max_pts)

            self._criteria_layout.addWidget(row_w)

        self._criteria_layout.addStretch()
        self._update_score()

    def _update_score(self) -> None:
        total  = sum(mp for _, mp in self._spinboxes.values())
        earned = sum(sb.value() for sb, _ in self._spinboxes.values())
        pct    = int(100 * earned / total) if total else 0
        self._score_label.setText(f"Score: {earned} / {total}  ({pct}%)")

    def _save_grade(self) -> None:
        if self._current_test is None:
            return
        grades_dir = self._current_test / "grades"
        grades_dir.mkdir(exist_ok=True)

        result: dict = {
            "test":      self._current_test.name,
            "timestamp": datetime.now().isoformat(),
            "criteria":  {},
            "notes":     self._notes.toPlainText().strip(),
        }
        for crit in self._criteria:
            cid = crit["id"]
            if crit.get("type") == "pass_fail":
                result["criteria"][cid] = {
                    "type":   "pass_fail",
                    "passed": self._checkboxes[cid].isChecked(),
                }
            else:
                sb, max_pts = self._spinboxes[cid]
                result["criteria"][cid] = {
                    "type":  "scored",
                    "score": sb.value(),
                    "max":   max_pts,
                }

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = grades_dir / f"grade_{stamp}.json"
        out_path.write_text(
            json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        QMessageBox.information(self, "Saved", f"Grade saved to:\n{out_path}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ConversationDashboard(QMainWindow):
    def __init__(self, messages: list[ParsedMessage], stats: ConversationStats):
        super().__init__()
        self.setWindowTitle("Claude Code — Conversation Review")
        self.resize(1440, 920)
        self.setMinimumSize(QSize(800, 500))

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = QToolBar("Actions")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            "QToolBar{background:#F9FAFB;border-bottom:1px solid #E5E7EB;spacing:6px;}"
        )
        grade_btn = QPushButton("Grade")
        grade_btn.setStyleSheet(
            "QPushButton{background:#2563EB;color:white;font-weight:bold;"
            "border-radius:5px;padding:4px 14px;font-size:11px;}"
            "QPushButton:hover{background:#1D4ED8;}"
        )
        grade_btn.clicked.connect(self._open_grading)
        toolbar.addWidget(grade_btn)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{background:{WINDOW_BG};border:none;}}")

        container = QWidget()
        container.setStyleSheet(f"background-color:{WINDOW_BG};")
        conv_layout = QVBoxLayout(container)
        conv_layout.setContentsMargins(24, 24, 24, 24)
        conv_layout.setSpacing(14)

        for msg in messages:
            if isinstance(msg, ParsedMicroscopeLog):
                widget: QWidget = LogBubble(msg)
            else:
                widget = MessageBubble(msg)

            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)

            if isinstance(msg, ParsedUserMessage) and isinstance(msg.content, str):
                row.addWidget(widget, 3)   # 3/4 left
                row.addStretch(1)
            else:
                row.addStretch(1)
                row.addWidget(widget, 3)   # 3/4 right

            row_w = QWidget()
            row_w.setLayout(row)
            row_w.setStyleSheet(f"background-color:{WINDOW_BG};")
            conv_layout.addWidget(row_w)

        conv_layout.addStretch()
        scroll.setWidget(container)
        splitter.addWidget(scroll)

        stats_panel = StatsPanel(stats)
        splitter.addWidget(stats_panel)

        splitter.setSizes([1110, 330])
        splitter.setCollapsible(0, False)

        self._grading_window: GradingWindow | None = None

    def _open_grading(self) -> None:
        if self._grading_window is None or not self._grading_window.isVisible():
            self._grading_window = GradingWindow(parent=self)
            self._grading_window.show()
        else:
            self._grading_window.raise_()
            self._grading_window.activateWindow()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_dashboard(messages: list[ParsedMessage], stats: ConversationStats) -> None:
    """Launch the PyQt6 dashboard. Blocks until the window is closed."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = ConversationDashboard(messages, stats)
    window.show()
    sys.exit(app.exec())
