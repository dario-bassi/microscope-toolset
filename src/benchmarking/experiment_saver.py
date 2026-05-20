"""Save a conversation slice from the active Claude Code session.

Usage (from the Claude Code prompt with ! prefix):

    Start an experiment:
        ! python -m src.benchmarking.experiment_saver start "cell_tracking"

    End and save it:
        ! python -m src.benchmarking.experiment_saver end

    List saved experiments:
        ! python -m src.benchmarking.experiment_saver list

The module tracks how many lines were in the session JSONL when the experiment
started and, on end, extracts only the lines added since then into a new file
under src/benchmarking/experiments/.

Each saved experiment folder contains:
  conversation.jsonl       — main session lines captured during the experiment
  session_data/            — copy of the session subdirectory, which includes:
    tool-results/<id>.txt  — large tool outputs offloaded from the JSONL
    subagents/<id>.jsonl   — each spawned subagent's full conversation
    subagents/<id>.meta.json — subagent type and description metadata
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR = DEFAULT_EXPERIMENTS_DIR  # backward-compat alias
MARKER_FILE = Path(".experiment_marker.json")


# ---------------------------------------------------------------------------
# Locating the Claude Code session file
# ---------------------------------------------------------------------------


def _claude_project_dir() -> Path | None:
    """Return ~/.claude/projects/<hash>/ for the current working directory."""
    cwd = Path.cwd()
    # Claude Code names the folder by replacing path separators and colons with '-'
    raw = str(cwd).replace(":", "-").replace("\\", "-").replace("/", "-")
    folder_name = raw.lstrip("-")

    candidates = Path.home() / ".claude" / "projects"
    exact = candidates / folder_name
    if exact.exists():
        return exact

    # Fallback: most recently touched project directory
    if candidates.exists():
        dirs = sorted(
            (d for d in candidates.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if dirs:
            return dirs[0]
    return None


def _latest_session_file(project_dir: Path) -> Path | None:
    """Return the most recently modified JSONL session file."""
    files = sorted(
        project_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _session_data_dir(session_file: Path) -> Path | None:
    """Return the <session-uuid>/ subdirectory next to the JSONL, if it exists."""
    candidate = session_file.parent / session_file.stem
    return candidate if candidate.is_dir() else None


def _line_count(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_experiment(
    name: str | None = None, base_dir: Path | str | None = None
) -> tuple[str, Path]:
    """Record the start of an experiment and return (name, workspace_dir).

    Creates the experiment folder and a workspace/ subdirectory immediately so
    the agent can start saving files there. Writes a `.experiment_marker.json`
    file with the folder paths and the current JSONL line offset.

    Args:
        name: optional experiment name; auto-generated if omitted.
        base_dir: parent directory for all experiments; defaults to
            ``DEFAULT_EXPERIMENTS_DIR`` (project root / "experiments").
    """
    if name is None:
        name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiments_dir = Path(base_dir) if base_dir else EXPERIMENTS_DIR
    # Pre-create the experiment folder so the agent has a workspace from the start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_").replace("/", "-")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = experiments_dir / f"{safe_name}_{timestamp}"
    workspace_dir = exp_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    project_dir = _claude_project_dir()
    session_file = _latest_session_file(project_dir) if project_dir else None
    line_count = _line_count(session_file) if session_file else 0
    session_data = _session_data_dir(session_file) if session_file else None

    marker = {
        "experiment_name": name,
        "start_time": datetime.now(UTC).isoformat(),
        "session_file": str(session_file) if session_file else None,
        "session_data_dir": str(session_data) if session_data else None,
        "start_line": line_count,
        "exp_dir": str(exp_dir),
        "workspace_dir": str(workspace_dir),
    }
    MARKER_FILE.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    print(f"[experiment_saver] Started '{name}'")
    print(f"  Experiment dir   : {exp_dir}")
    print(f"  Workspace        : {workspace_dir}")
    print(f"  Session file     : {session_file}")
    print(f"  Starting at      : line {line_count}")
    return name, workspace_dir


def end_experiment(output_dir: Path | str | None = None) -> Path:
    """Extract conversation lines since start_experiment() and save to an experiment folder.

    The experiment folder contains:
      conversation.jsonl  — main session lines captured during the experiment
      session_data/       — copy of the session subdirectory (tool-results + subagents)

    Returns the path to the experiment folder.
    Raises FileNotFoundError if no experiment has been started.
    """
    if not MARKER_FILE.exists():
        raise FileNotFoundError("No active experiment. Run start_experiment() first.")

    marker = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    name = marker["experiment_name"]
    start_line: int = marker["start_line"]
    session_path = Path(marker["session_file"])

    if not session_path.exists():
        raise FileNotFoundError(
            f"Session file not found: {session_path}\n"
            "The session may have been cleaned up by Claude Code."
        )

    with open(session_path, encoding="utf-8") as f:
        all_lines = f.readlines()

    exp_lines = all_lines[start_line:]
    if not exp_lines:
        print("[experiment_saver] Warning: no new lines captured since start.")

    # Use the pre-created experiment folder from the marker when available;
    # fall back to creating a new timestamped folder for legacy markers.
    if "exp_dir" in marker:
        base_dir = Path(output_dir) if output_dir else None
        exp_dir = (
            Path(marker["exp_dir"]) if base_dir is None else base_dir / Path(marker["exp_dir"]).name
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_").replace("/", "-")
        base_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR
        exp_dir = base_dir / f"{safe_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Save conversation slice
    conv_path = exp_dir / "conversation.jsonl"
    with open(conv_path, "w", encoding="utf-8") as f:
        f.writelines(exp_lines)
    print(f"[experiment_saver] conversation.jsonl  — {len(exp_lines)} lines")

    # Copy session subdirectory (tool-results + subagents)
    # Prefer the path recorded at start; fall back to deriving it now.
    recorded = marker.get("session_data_dir")
    session_data = Path(recorded) if recorded and Path(recorded).is_dir() else None
    if session_data is None:
        session_data = _session_data_dir(session_path)

    if session_data and session_data.is_dir():
        dest = exp_dir / "session_data"
        shutil.copytree(session_data, dest)
        n_tool = (
            len(list((dest / "tool-results").glob("*.txt")))
            if (dest / "tool-results").exists()
            else 0
        )
        n_agent = (
            len(list((dest / "subagents").glob("*.jsonl"))) if (dest / "subagents").exists() else 0
        )
        print(
            f"[experiment_saver] session_data/       — {n_tool} tool-result file(s), {n_agent} subagent(s)"
        )
    else:
        print("[experiment_saver] session_data/       — (no session subdirectory found)")

    MARKER_FILE.unlink()

    print(f"\n[experiment_saver] Saved → {exp_dir}")
    return exp_dir


def list_experiments(experiments_dir: Path | str | None = None) -> list[Path]:
    """List all saved experiment folders, newest first.

    Returns paths to experiment directories (each containing conversation.jsonl).
    """
    d = Path(experiments_dir) if experiments_dir else EXPERIMENTS_DIR
    if not d.exists():
        return []
    folders = sorted(
        (p for p in d.iterdir() if p.is_dir() and (p / "conversation.jsonl").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return folders


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == "start":
        name = args[1] if len(args) > 1 else None
        _, workspace = start_experiment(name)
        print(f"  Workspace ready at: {workspace}")

    elif cmd == "end":
        try:
            exp_dir = end_experiment()
            print("\nOpen in dashboard:")
            print(
                f"  python -m src.benchmarking.view_experiment \"{exp_dir / 'conversation.jsonl'}\""
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "list":
        folders = list_experiments()
        if not folders:
            print(f"No experiments found in {EXPERIMENTS_DIR}")
        else:
            print(f"Experiments in {EXPERIMENTS_DIR}:")
            for folder in folders:
                conv = folder / "conversation.jsonl"
                size_kb = conv.stat().st_size / 1024
                has_agents = (folder / "session_data" / "subagents").exists()
                has_tools = (folder / "session_data" / "tool-results").exists()
                extras = []
                if has_agents:
                    n = len(list((folder / "session_data" / "subagents").glob("*.jsonl")))
                    extras.append(f"{n} subagent(s)")
                if has_tools:
                    n = len(list((folder / "session_data" / "tool-results").glob("*.txt")))
                    extras.append(f"{n} tool-result(s)")
                suffix = f"  [{', '.join(extras)}]" if extras else ""
                print(f"  {folder.name}  ({size_kb:.1f} KB){suffix}")

    elif cmd == "status":
        if MARKER_FILE.exists():
            m = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
            print(f"Active experiment: '{m['experiment_name']}'")
            print(f"  Started at : {m['start_time']}")
            print(f"  Start line : {m['start_line']}")
            session = Path(m["session_file"])
            current = _line_count(session)
            print(f"  Lines so far: {current - m['start_line']} new lines captured")
        else:
            print("No active experiment.")

    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print("Commands: start [name] | end | list | status")
        sys.exit(1)


if __name__ == "__main__":
    _main()
