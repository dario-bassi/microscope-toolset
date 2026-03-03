# pymmcore-proxy / run_events Migration Notes

> **Read this when reusing or adapting code from old scratch/ challenge scripts.**

---

## Breaking changes (Feb 2026)

### 1. `run_events()` now delegates to `core.mda.run()`

Old scratch scripts may contain patterns like this:

```python
# OLD — run_events() was a manual for-loop with raw snapImage()
# Still works (same signature), but now correctly uses the MDA engine
results = run_events(core, my_generator(), on_frame=callback)
```

**This still works.** The signature of `run_events()` is unchanged. However, the underlying behaviour changed significantly:

| Old | New |
|-----|-----|
| Manual `for event in events: core.snapImage()` loop | `core.mda.run(events)` — real MDA engine |
| No hardware sequencing | Hardware sequencing (bursts compatible events) |
| No pause/cancel support | `core.mda.toggle_pause()`, `core.mda.cancel()` work |
| Timing handled manually with `time.sleep` | MDA engine handles `min_start_time` |

### 2. `execute_mda()` removed

Old code may call:
```python
from src.workflows.mda import execute_mda
execute_mda(core, events, on_frame=callback)
```

**Replace with `run_events()`** — they are now identical:
```python
from src.hardware.core import run_events
run_events(core, events, on_frame=callback)
```

### 3. `run_mda()` removed from `src.hardware.core`

Old code may call:
```python
from src.hardware.core import run_mda
run_mda(core, my_generator, on_frame_cb=callback)
```

**Replace with:**
```python
run_events(core, my_generator(), on_frame=callback)
```

### 4. `run_bacteria_trap()` removed from `src.workflows.bacteria_trap`

Old code may call:
```python
from src.workflows.bacteria_trap import run_bacteria_trap
result = run_bacteria_trap(core, cx, cy, radius, ...)
```

**Replace with `run_bacteria_trap_mda()`** — same purpose, proper MDA:
```python
from src.workflows.bacteria_trap import run_bacteria_trap_mda
result = run_bacteria_trap_mda(core, cx, cy, radius, ...)
```

### 5. `metadata={'properties': {...}}` pattern no longer works

Old code may set per-event device properties via metadata:
```python
# OLD — only worked with the old manual run_events() loop
MDAEvent(metadata={'properties': {'Camera.Gain': 4.0}})
```

**Replace with the native `event.properties` field:**
```python
# NEW — handled natively by the MDA engine
MDAEvent(properties=[('Camera', 'Gain', '4.0')])
```

---

## What has NOT changed

- `run_events(core, events, on_frame=callback)` — same call signature
- `MDASequence` + `run_events()` patterns — unchanged
- Generator-based adaptive acquisition — unchanged
- All `on_frame(img, event)` callbacks — unchanged
