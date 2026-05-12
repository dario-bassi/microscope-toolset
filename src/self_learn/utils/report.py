"""Experiment report generator.

Produces structured summaries from ExperimentLog data with
measurement statistics, phase comparisons, and formatted output.

Functions:
    generate_report      -- Build structured report from ExperimentLog
    format_markdown      -- Render report as markdown text
    measurement_table    -- Tabular summary with statistics per metric
    phase_comparison     -- Compare a metric across experiment phases
    experiment_timeline  -- Chronological event summary
"""

import numpy as np
from collections import defaultdict


def generate_report(log):
    """Build a structured report from an ExperimentLog.

    Args:
        log: ExperimentLog instance (or dict from log.to_dict()).

    Returns:
        dict with sections:
            experiment_id: str
            description: str
            start_time: str
            config: dict of configuration parameters
            phases: list of phase names in order
            acquisitions: list of acquisition records
            measurements: dict mapping name → list of values
            measurement_stats: dict mapping name → {mean, std, min, max, n, units}
            decisions: list of decision descriptions
            devices: list of device change records
            timeline: list of (elapsed_s, event_type, summary) tuples
            n_entries: int
    """
    entries = _get_entries(log)
    info = _get_info(log)

    config = {}
    phases = []
    acquisitions = []
    measurements = defaultdict(list)
    measurement_units = {}
    decisions = []
    devices = []
    timeline = []

    for e in entries:
        etype = e["type"]
        data = e.get("data", {})
        elapsed = e.get("elapsed_s", 0)

        if etype == "config":
            config.update(data)
            timeline.append((elapsed, "config", _summarize_config(data)))
        elif etype == "phase_change":
            phase = data.get("phase", "unknown")
            phases.append(phase)
            timeline.append((elapsed, "phase", f"→ {phase}"))
        elif etype == "acquisition":
            acquisitions.append(data)
            label = data.get("label", "?")
            ch = data.get("channel", "")
            timeline.append((elapsed, "acquisition", f"{label} ({ch})" if ch else label))
        elif etype == "measurement":
            name = data.get("name", "?")
            value = data.get("value", 0)
            units = data.get("units", "")
            measurements[name].append(value)
            if units:
                measurement_units[name] = units
            timeline.append((elapsed, "measurement", f"{name}={value} {units}".strip()))
        elif etype == "decision":
            desc = data.get("description", "")
            decisions.append(desc)
            timeline.append((elapsed, "decision", desc))
        elif etype == "device_change":
            devices.append(data)
            dev = data.get("device", "?")
            state = data.get("state", "?")
            timeline.append((elapsed, "device", f"{dev} → {state}"))
        else:
            summary = str(data)[:60] if data else etype
            timeline.append((elapsed, etype, summary))

    # Compute statistics
    measurement_stats = {}
    for name, values in measurements.items():
        arr = np.array(values, dtype=np.float64)
        measurement_stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
            "units": measurement_units.get(name, ""),
        }

    return {
        "experiment_id": info.get("experiment_id", ""),
        "description": info.get("description", ""),
        "start_time": info.get("start_time", ""),
        "config": config,
        "phases": phases,
        "acquisitions": acquisitions,
        "measurements": dict(measurements),
        "measurement_stats": measurement_stats,
        "decisions": decisions,
        "devices": devices,
        "timeline": timeline,
        "n_entries": len(entries),
    }


def format_markdown(report):
    """Render a report dict as markdown text.

    Args:
        report: dict from generate_report().

    Returns:
        str: Markdown-formatted report.
    """
    lines = []
    lines.append(f"# {report['experiment_id']}")
    if report.get("description"):
        lines.append(f"\n{report['description']}")
    if report.get("start_time"):
        lines.append(f"\n*Started: {report['start_time']}*")

    # Configuration
    if report.get("config"):
        lines.append("\n## Configuration\n")
        for k, v in report["config"].items():
            lines.append(f"- **{k}**: {v}")

    # Phases
    if report.get("phases"):
        lines.append(f"\n## Phases\n")
        lines.append(" → ".join(report["phases"]))

    # Measurements
    if report.get("measurement_stats"):
        lines.append("\n## Measurements\n")
        lines.append("| Metric | Mean | Std | Min | Max | N | Units |")
        lines.append("|--------|------|-----|-----|-----|---|-------|")
        for name, stats in report["measurement_stats"].items():
            lines.append(
                f"| {name} | {stats['mean']:.3g} | {stats['std']:.3g} | "
                f"{stats['min']:.3g} | {stats['max']:.3g} | {stats['n']} | "
                f"{stats['units']} |"
            )

    # Decisions
    if report.get("decisions"):
        lines.append("\n## Decisions\n")
        for d in report["decisions"]:
            lines.append(f"1. {d}")

    # Device changes
    if report.get("devices"):
        lines.append("\n## Device Changes\n")
        for d in report["devices"]:
            lines.append(f"- {d.get('device', '?')}: {d.get('state', '?')}")

    # Acquisitions summary
    if report.get("acquisitions"):
        lines.append(f"\n## Acquisitions ({len(report['acquisitions'])} total)\n")
        channels = defaultdict(int)
        for a in report["acquisitions"]:
            ch = a.get("channel", a.get("label", "unknown"))
            channels[ch] += 1
        for ch, count in channels.items():
            lines.append(f"- {ch}: {count} frames")

    return "\n".join(lines)


def measurement_table(log):
    """Build a tabular summary of all measurements with statistics.

    Args:
        log: ExperimentLog instance or dict.

    Returns:
        dict mapping metric_name → {values, mean, std, min, max, n, units}.
    """
    entries = _get_entries(log)
    grouped = defaultdict(list)
    units_map = {}

    for e in entries:
        if e["type"] == "measurement":
            data = e.get("data", {})
            name = data.get("name", "?")
            grouped[name].append(data.get("value", 0))
            if data.get("units"):
                units_map[name] = data["units"]

    result = {}
    for name, values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        result[name] = {
            "values": values,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
            "units": units_map.get(name, ""),
        }
    return result


def phase_comparison(log, metric_name, phases=None):
    """Compare a metric across experiment phases.

    Args:
        log: ExperimentLog instance or dict.
        metric_name: Name of the measurement to compare.
        phases: Optional list of phase names to include. If None, uses all.

    Returns:
        dict mapping phase → {mean, std, n, values}.
        Empty dict if metric not found.
    """
    entries = _get_entries(log)
    phase_values = defaultdict(list)

    for e in entries:
        if e["type"] == "measurement" and e["data"].get("name") == metric_name:
            phase = e.get("phase", "unknown")
            if phases is None or phase in phases:
                phase_values[phase].append(e["data"].get("value", 0))

    result = {}
    for phase, values in phase_values.items():
        arr = np.array(values, dtype=np.float64)
        result[phase] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "n": len(arr),
            "values": values,
        }
    return result


def experiment_timeline(log, max_entries=50):
    """Produce a chronological event summary.

    Args:
        log: ExperimentLog instance or dict.
        max_entries: Maximum number of entries to return.

    Returns:
        list of dicts with keys: elapsed_s, type, phase, summary.
    """
    entries = _get_entries(log)
    timeline = []

    for e in entries[:max_entries]:
        data = e.get("data", {})
        etype = e["type"]
        phase = e.get("phase", "")
        elapsed = e.get("elapsed_s", 0)

        if etype == "config":
            summary = _summarize_config(data)
        elif etype == "phase_change":
            summary = f"→ {data.get('phase', '?')}"
        elif etype == "acquisition":
            label = data.get("label", "?")
            ch = data.get("channel", "")
            summary = f"{label} ({ch})" if ch else label
        elif etype == "measurement":
            name = data.get("name", "?")
            val = data.get("value", "?")
            units = data.get("units", "")
            summary = f"{name} = {val} {units}".strip()
        elif etype == "decision":
            summary = data.get("description", "?")
        elif etype == "device_change":
            summary = f"{data.get('device', '?')} → {data.get('state', '?')}"
        else:
            summary = str(data)[:60] if data else etype

        timeline.append({
            "elapsed_s": elapsed,
            "type": etype,
            "phase": phase,
            "summary": summary,
        })

    return timeline


def _get_entries(log):
    """Extract entries list from ExperimentLog or dict."""
    if isinstance(log, dict):
        return log.get("entries", [])
    return getattr(log, "entries", [])


def _get_info(log):
    """Extract metadata from ExperimentLog or dict."""
    if isinstance(log, dict):
        return log
    return {
        "experiment_id": getattr(log, "experiment_id", ""),
        "description": getattr(log, "description", ""),
        "start_time": getattr(log, "start_iso", ""),
    }


def _summarize_config(data):
    """Compact string summary of config dict."""
    parts = [f"{k}={v}" for k, v in data.items()]
    return ", ".join(parts[:5])
