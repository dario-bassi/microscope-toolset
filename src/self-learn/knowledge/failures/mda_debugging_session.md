# MDASequence Configuration & Execution Issues

## Session: Cell Segmentation & Tracking (1-minute timelapse, 200ms intervals)

Investigated 6 problems encountered while implementing cell segmentation with live tracking. Key finding: **Always query microscope settings before building MDA configurations**.

---

## Problem 1: Module Import Failures

### Symptom
```
ModuleNotFoundError: No module named 'src.hardware.core'
```

### Root Cause
Project structure moved all code under `src/self-learn/`, breaking old imports from `src.analysis`, `src.detection`, etc.

### Solution
✓ Import from: `from src.self_learn.hardware.core import ...`
✓ Updated all import paths in CLAUDE.md

---

## Problem 2: MDASequence Channel Configuration Mismatch

### Symptom
```
ValueError: Configuration group "Channel" or its preset "all-channel" does not exist
```

### Root Cause
**MDA runner looks for `group="Channel"` (hardcoded default)**, but this simulation uses `group="Fake"` with presets: `all-channel`, `membrane-channel`, `nucleus-channel`.

### Solution
✓ **Always query and specify channel group explicitly:**
```python
from useq import Channel

current_group = mmc.getChannelGroup()  # "Fake"
available = mmc.getAvailableConfigs(current_group)

seq = MDASequence(
    time_plan={"loops": 300, "interval": 0.2},
    channels=[Channel(
        group=current_group,  # ← CRITICAL
        config=available[0],
        exposure=50
    )]
)
```

### Lesson
- Always query `getChannelGroup()` before building sequences
- Channel defaults to `group='Channel'` — override with actual microscope config
- This issue is simulation-specific; real hardware uses "Channel" exactly

---

## Problem 3: MDASequence Without Channels Returns None

### Symptom
```
TypeError: 'NoneType' object is not iterable
```

### Root Cause
`mmc.mda.run()` returns `None` when no channels specified (silent validation failure)

### Solution
✓ Always specify at least one channel in MDASequence

---

## Problem 4: File Path Issues

### Issue 4a: Unix paths
```python
path = "/tmp/cell_tracks.npz"  # ✗ FileNotFoundError on Windows
```

### Issue 4b: __file__ undefined
```python
output_dir = os.path.dirname(os.path.abspath(__file__))  # ✗ NameError
```

### Solution
```python
import os
output_path = os.path.join(os.getcwd(), "cell_tracks.npz")  # ✓ Cross-platform
```

### Lesson
- Use `os.path.join()` + `os.getcwd()` for all paths
- Never hardcode separators or `__file__`

---

## Problem 5: Timing Overhead

### Symptom
Acquisition took 153s instead of 60s (2.5× overhead)

### Root Cause
- Cell detection per frame
- Hungarian matching per frame
- `time.sleep()` jitter
- Simulation overhead

### Impact
✓ All 300 frames acquired successfully
⚠️ Plan: `actual_time = theoretical_time × 2.5` for loop-based acquisition

### Solution
Use MDASequence for better timing via hardware infrastructure

---

## Problem 6: Incomplete Tracks Data

### Symptom
Added only 3 track points instead of all 1,268 to napari

### Root Cause
Used partial data subset instead of full array

### Solution
```python
data = np.load("cell_tracks.npz")
all_tracks = data['tracks']  # All 1,268 points
viewer_add_tracks(all_tracks)  # ✓ Complete trajectories
```

---

## Checklist for Future Acquisitions

- [ ] Query: `getChannelGroup()`, `getAvailableConfigs()`
- [ ] Build: Channel with explicit `group=` parameter
- [ ] Paths: Use `os.path.join(os.getcwd(), ...)`
- [ ] Timing: Plan 2-3× overhead for processing-heavy loops
- [ ] Data: Always use complete datasets for visualization
