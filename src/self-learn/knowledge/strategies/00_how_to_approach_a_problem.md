# How to Approach a Problem

Follow these 7 steps for every challenge. Skipping steps is the #1 source of errors.

## Step 1: Understand the Question
- **What measurement?** Count, distance, intensity ratio, speed, classification?
- **What units?** Pixels or micrometers? Per-cell or per-field?
- **What output format?** Single number, list, dictionary, JSON?

Read the question twice before touching the microscope.

## Step 2: Discover the Instrument
```python
groups = core.getAvailableConfigGroups()          # "Channel" or "Fake"
channels = core.getAvailableConfigs(group)        # ["BF", "GFP", "nucleus", ...]
objectives = core.getAvailableConfigs("Objective") # ["10", "20", "40"]
pixel_size = core.getPixelSizeUm()
slm = core.getSLMDevice()                         # "" if no SLM
# Check state devices
for dev in core.getLoadedDevices():
    try:
        labels = core.getStateLabels(dev)
        if labels: print(f'{dev}: {list(labels)}')
    except: pass
```
Discover: channels, objectives, pixel size, SLM, state devices (Temperature, Anesthesia, etc.).

**CRITICAL: Read channel descriptions in the challenge notes.** Snap a test image from
EACH channel and LOOK at it before committing to an analysis channel. The channel name
(e.g., "membrane-channel") may NOT match the fluorophore you expect. This is the #1
source of costly errors (ch421: 5/10 → searched wrong channel for vasculature).

## Step 3: Find the Sample (10x Overview First)
```python
from useq import MDAEvent
from src.hardware.core import run_events
events = [MDAEvent(channel={"config": "brightfield", "group": grp})]
overview = run_events(core, events)
img_10x = overview[0][0]
```
- Count total objects at 10x — this is your ground truth cell count
- Record world positions for high-mag visits
- Save to /tmp/ and visually inspect

**Critical**: If the task needs BOTH counting AND subcellular detail, do low-mag overview first.

## Step 4: Design Acquisition (Backward from Analysis)
Choose the lowest magnification that resolves the features. BF before fluorescence.

**Use MDA for all multi-frame acquisitions** (NEVER manual snap() loops):
- Timelapse → `MDASequence(time_plan={"loops": N, "interval": dt})`
- Z-stack → `MDASequence(z_plan={"range": R, "step": S}, channels=[...])`
- Multi-position → `MDASequence(stage_positions=[...])`
- Adaptive → Generator yielding MDAEvents with per-frame analysis callback

Single-frame acquisitions can still use `snap()`.

## Step 5: Test on a Few Frames
```python
core.snapImage()
test_img = core.getImage()
from skimage.measure import label, regionprops
mask = test_img > threshold
props = regionprops(label(mask))
print(f"Detected {len(props)} objects")
```
Save overlay to /tmp/ and verify visually. Fix detection before running full experiment.

## Step 6: Execute
**Fixed acquisition:**
```python
from useq import MDASequence
seq = MDASequence(channels=[...], time_plan={"loops": N})
results = run_events(core, list(seq), on_frame=callback)
```

**Adaptive / closed-loop** — use a generator:
```python
def my_generator():
    for i in range(max_steps):
        yield MDAEvent(channel={"config": "BF", "group": "Fake"})
        # Decision logic between yields
results = run_events(core, my_generator(), on_frame=analyze_frame)
```

**SLM**: `MDAEvent(slm_image=SLMImage(data=mask, device="SLM"))`

## Step 7: Verify
- **Visual**: overlay detections on image, inspect
- **Numerical**: is the count reasonable? Speed positive?
- **Units**: pixels → micrometers where needed
- **Format**: int vs float, dict keys match expected output

Common errors: numpy int64 not JSON-serializable, pixel coords instead of world coords.

| Step | Action |
|------|--------|
| 1. Understand | Read question twice |
| 2. Discover | Query hardware, snap ALL channels, verify fluorophores |
| 3. Find sample | 10x overview |
| 4. Design | Choose MDA pattern |
| 5. Test | Validate detection on 1 frame |
| 6. Execute | Run MDA acquisition |
| 7. Verify | Sanity-check results |
