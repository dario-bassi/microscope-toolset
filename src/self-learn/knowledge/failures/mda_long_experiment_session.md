# MDA Long Experiment — Debugging Session

## Session: 60h Optogenetic Tracking Experiment (2026-03-13 to 2026-03-16)

Running a 60-hour closed-loop optogenetic experiment revealed multiple issues with
data persistence, threading, and the MCP execution model.

---

## Problem 1: run_mda_with_feedback Signature Confusion

### Symptom
```
TypeError: Execute.__init__.<locals>.<lambda>() got multiple values for argument 'on_frame'
```

### Root Cause
The actual function in `mda_helpers.py` has signature `run_mda_with_feedback(mmc, events, on_frame)`.
But in the MCP sandbox, it is **pre-bound** via a lambda in `execute.py`:
```python
self.namespace["run_mda_with_feedback"] = lambda events, on_frame=None: run_mda_with_feedback(mmc, events, on_frame)
```
So in sandbox code, the signature is `run_mda_with_feedback(events, on_frame=callback)` — NO `mmc`.
Passing `mmc` as first arg causes `events` to receive `mmc` and `on_frame` gets the generator,
then the keyword `on_frame=` conflicts.

### Solution
In MCP sandbox: `run_mda_with_feedback(my_generator(), on_frame=callback)`
Outside sandbox: `run_mda_with_feedback(mmc, my_generator(), on_frame=callback)`

### Lesson
**Check how functions are exposed in the sandbox.** The sandbox may wrap functions with
pre-bound arguments. This mistake was made TWICE across sessions because CLAUDE.md
itself had the wrong signature documented.

---

## Problem 2: run_mda_with_feedback Fails in Daemon Threads

### Symptom
Experiment started in `threading.Thread(target=run_experiment, daemon=True)` completed
instantly with 0 frames captured. Duration: 0.09 seconds.

### Root Cause
The MDA engine in pymmcore-plus cannot run from a daemon thread. The `mmc.run_mda()`
call returns immediately without executing any events.

### Solution
Call `run_mda_with_feedback()` directly in the MCP `execute_python_code` context (not
in a background thread). The MCP tool will block for the experiment duration, but the
Python namespace survives after timeout.

### Lesson
**Never wrap run_mda_with_feedback in a daemon thread.** The MDA engine needs to run
on the proper thread context. For long experiments, accept the blocking behavior.

---

## Problem 3: MCP Timeout Kills Save Code

### Symptom
60h experiment completed successfully (MDA finished at 08:08 Monday). But the save
code placed AFTER `run_mda_with_feedback()` was killed when the MCP tool timed out
at exactly 216000s. Only ~half of mScarlet3 was written to disk.

### Root Cause
The `execute_python_code` MCP tool has a timeout. When the MDA blocks for 60h, the
timeout fires at the same moment the MDA completes, killing the post-MDA save code
mid-execution.

### Solution (applied retroactively)
The Python namespace (`state` dict with all frames) survived the timeout. A follow-up
`execute_python_code` call recovered all 7200+7200+14400 frames from memory and saved
them to disk.

### Better Solution (for future experiments)
Save data **incrementally** in the `on_frame` callback:
```python
def on_frame(image, event, metadata):
    # Save each frame immediately to disk
    frame_idx = state["frame_count"]
    tifffile.imwrite(f"{out_dir}/frame_{frame_idx:06d}.tif", image)
    state["frame_count"] += 1

    # Also save tracking log periodically
    if frame_idx % 100 == 0:
        with open(f"{out_dir}/tracking_log.json", "w") as f:
            json.dump(state["log"], f)
```

### Lesson
**Never rely on post-experiment save code for long experiments.** Save incrementally
during acquisition. Also use `bigtiff=True` for stacks >2GB.

---

## Problem 4: TIFF 4GB Size Limit

### Symptom
```
error: 'I' format requires 0 <= number <= 4294967295
```

### Root Cause
Standard TIFF uses 32-bit offsets, limiting files to ~4GB. A 7200-frame uint16
timelapse (1024x1024) is ~15GB.

### Solution
Use `tifffile.imwrite(..., bigtiff=True)` for any timelapse expected to exceed 2GB.

---

## Problem 5: VS Code Freezing

### Symptom
VS Code became unresponsive during the data save phase, requiring a force close.

### Root Cause
~30GB of image data in memory (7200+7200+14400 frames × 1024×1024 × 2 bytes) plus
the serialization overhead of writing large TIFF files.

### Solution
Save incrementally to avoid accumulating all frames in memory. For the 60h experiment
with 14400 stim frames, this would have kept memory usage constant instead of growing
to 30GB.

---

## Problem 6: DMD setSLMExposure Not Used

### Symptom
Manual snap attempts showed no signal even with DMD white mask uploaded.

### Root Cause
Forgot to call `mmc.setSLMExposure('Mosaic3', 5000)` before `displaySLMImage()`.
Without it, the DMD may turn off before the camera exposure completes.

### Solution
For manual snaps (outside MDA), always set SLM exposure:
```python
mmc.setSLMExposure(slm_device, 5000)  # ms — keep DMD on for 5 seconds
mmc.setSLMImage(slm_device, white_mask)
mmc.displaySLMImage(slm_device)
# Now snap within the 5s window
```
When using `SLMImage` in `MDAEvent`, the engine handles this automatically.

---

## Summary: Long Experiment Checklist

1. ✅ Use `run_mda_with_feedback(events, on_frame=callback)` — NO `mmc` in sandbox
2. ✅ Do NOT wrap in a daemon thread — run directly
3. ✅ Save frames incrementally in `on_frame` callback
4. ✅ Use `bigtiff=True` for large stacks
5. ✅ Save tracking log periodically (every 100 frames)
6. ✅ Use `setSLMExposure()` for manual snaps, `SLMImage` for MDA events
7. ✅ Accept that the MCP tool will block — data survives in namespace after timeout
8. ✅ For DMD imaging: always activate DMD (white mask) for ALL channels, not just stim
