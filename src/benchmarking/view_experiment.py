"""Launch the dashboard for a saved experiment JSONL file.

Usage:
    python -m src.benchmarking.view_experiment path/to/experiment.jsonl

    # Or pick interactively from the experiments/ folder:
    python -m src.benchmarking.view_experiment
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.benchmarking.dashboard import launch_dashboard
from src.benchmarking.experiment_saver import EXPERIMENTS_DIR, list_experiments
from src.benchmarking.review_conversation import read_file


def _pick_experiment() -> Path | None:
    """Prompt the user to choose from the experiments/ folder."""
    folders = list_experiments()
    if not folders:
        print(f"No experiments found in {EXPERIMENTS_DIR}")
        return None

    print("Available experiments:\n")
    for i, folder in enumerate(folders, 1):
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
        print(f"  [{i}] {folder.name}  ({size_kb:.1f} KB){suffix}")

    print()
    try:
        choice = int(input("Enter number to open (0 to cancel): "))
    except (ValueError, KeyboardInterrupt):
        return None

    if choice == 0 or choice > len(folders):
        return None
    return folders[choice - 1] / "conversation.jsonl"


def _resolve_path(arg: str) -> Path:
    """Accept either an experiment folder or a direct .jsonl path."""
    p = Path(arg)
    if p.is_dir() and (p / "conversation.jsonl").exists():
        return p / "conversation.jsonl"
    return p


def main() -> None:
    if len(sys.argv) > 1:
        path = _resolve_path(sys.argv[1])
    else:
        path = _pick_experiment()
        if path is None:
            sys.exit(0)

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {path} ...")
    messages, stats = read_file(str(path))
    stats.session_id = stats.session_id or path.stem
    launch_dashboard(messages, stats)


if __name__ == "__main__":
    main()
