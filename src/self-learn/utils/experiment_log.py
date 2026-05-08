"""Structured experiment logging for reproducibility.

Records every acquisition decision, measurement, and state change
during a microscopy experiment. Logs are JSON-serializable for
storage and replay.

Usage:
    log = ExperimentLog("ch426_cell_cycle")
    log.log_config(objective=20, pixel_size=0.5)
    log.log_acquisition("baseline", channel="nucleus-channel", exposure=50)
    log.log_measurement("cell_count", 32, units="cells")
    log.log_decision("arrest detected", {"g1_frac": 0.95})
    log.log_device("Perfusion", "Drug")
    log.save("/tmp/experiment.json")
"""

import json
import time
from datetime import datetime
from pathlib import Path


class ExperimentLog:
    """Structured experiment log with timestamped entries."""

    def __init__(self, experiment_id, description=""):
        self.experiment_id = experiment_id
        self.description = description
        self.start_time = time.time()
        self.start_iso = datetime.now().isoformat()
        self.entries = []
        self._phase = "setup"

    def set_phase(self, phase):
        """Set the current experiment phase (e.g., 'baseline', 'treatment', 'recovery')."""
        self._phase = phase
        self._add_entry("phase_change", {"phase": phase})

    def log_config(self, **kwargs):
        """Log microscope configuration."""
        self._add_entry("config", kwargs)

    def log_acquisition(self, label, **kwargs):
        """Log an image acquisition event."""
        self._add_entry("acquisition", {"label": label, **kwargs})

    def log_measurement(self, name, value, units="", metadata=None):
        """Log a quantitative measurement."""
        entry = {"name": name, "value": value, "units": units}
        if metadata:
            entry["metadata"] = metadata
        self._add_entry("measurement", entry)

    def log_decision(self, description, context=None):
        """Log an experimental decision and its rationale."""
        entry = {"description": description}
        if context:
            entry["context"] = context
        self._add_entry("decision", entry)

    def log_device(self, device, state):
        """Log a device state change."""
        self._add_entry("device_change", {"device": device, "state": state})

    def log_event(self, event_type, data=None):
        """Log a generic event."""
        self._add_entry(event_type, data or {})

    def _add_entry(self, entry_type, data):
        """Add a timestamped entry."""
        elapsed = time.time() - self.start_time
        self.entries.append({
            "type": entry_type,
            "phase": self._phase,
            "elapsed_s": round(elapsed, 3),
            "data": data,
        })

    def get_measurements(self, name=None):
        """Retrieve all measurements, optionally filtered by name."""
        results = []
        for e in self.entries:
            if e["type"] == "measurement":
                if name is None or e["data"]["name"] == name:
                    results.append(e["data"])
        return results

    def get_phase_entries(self, phase):
        """Get all entries from a specific phase."""
        return [e for e in self.entries if e["phase"] == phase]

    def summary(self):
        """Generate a text summary of the experiment."""
        lines = [
            f"Experiment: {self.experiment_id}",
            f"Started: {self.start_iso}",
            f"Description: {self.description}",
            f"Total entries: {len(self.entries)}",
        ]

        # Phase summary
        phases = []
        for e in self.entries:
            if e["type"] == "phase_change":
                phases.append(e["data"]["phase"])
        if phases:
            lines.append(f"Phases: {' → '.join(phases)}")

        # Measurement summary
        measurements = self.get_measurements()
        if measurements:
            lines.append(f"Measurements: {len(measurements)}")
            for m in measurements[:10]:
                val = m["value"]
                units = m.get("units", "")
                lines.append(f"  {m['name']}: {val} {units}")

        # Decision summary
        decisions = [e for e in self.entries if e["type"] == "decision"]
        if decisions:
            lines.append(f"Decisions: {len(decisions)}")
            for d in decisions[:5]:
                lines.append(f"  - {d['data']['description']}")

        return "\n".join(lines)

    def to_dict(self):
        """Serialize to a JSON-compatible dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "start_time": self.start_iso,
            "duration_s": round(time.time() - self.start_time, 1),
            "n_entries": len(self.entries),
            "entries": self.entries,
        }

    def save(self, path):
        """Save log to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return str(path)

    @classmethod
    def load(cls, path):
        """Load a saved experiment log."""
        with open(path) as f:
            data = json.load(f)
        log = cls(data["experiment_id"], data.get("description", ""))
        log.start_iso = data.get("start_time", "")
        log.entries = data.get("entries", [])
        return log
