# Cell Tracking with Experimental Discovery

## Overview

You will conduct a three-phase workflow:
1. **LEARN PHASE**: Capture and analyze sample images
2. **FEEDBACK PHASE**: User reviews your findings and provides guidance
3. **EXPERIMENT PHASE**: Execute the final tracking protocol

You have full access to the microscope hardware and MCP Tools. Use this to your advantage!

---

## PHASE 1: LEARN PHASE (Discovery & Analysis)

Your goal: Understand what's in the microscope.

### What You Should Do

1. **Capture sample images** 

2. **Analyze the first image to understand the scene**
   - How many objects are present?
   - What's their typical size in pixels?
   - How bright are they relative to background?
   - Is the background uniform or noisy?
   - What's the signal-to-noise ratio?

3. **Test detection approaches**
   - Try simple thresholding first
   - Try blob detection
   - Report: "Method X found N objects"
   - Show: object positions, sizes, confidence scores

4. **Evaluate object closest to center**
   - Is there an object within 10 microns of center?
   - How far is closest object from center?
   - Will tracking be feasible?

5. **Using Screenshots to Show Your Work**

    After detecting objects:

    1. Create visualization showing detections overlaid on image
    2. Take screenshot (use mcp_take_screenshot tool)
    3. Analyze screenshot to assess detection quality
    4. Report findings + visual confidence
    5. Display screenshot and ask user: "Does this look correct?"
    6. WAIT for user feedback before proceeding

    This gives user a fast way to approve or request changes.

    #### Visual Assessment Questions

    When analyzing your detection screenshot, consider:
    - Are circles/markers in right positions?
    - Are there obvious false positives (noise marked as objects)?
    - Are all real objects detected?
    - Is there anything suspicious or wrong?
    - How confident am I (HIGH/MEDIUM/LOW)?

    Report your assessment explicitly:
    ```
    VISUAL ASSESSMENT:
    - 5 objects detected, all well-separated ✓
    - No obvious false positives ✓
    - Closest object at (200, 250), distance from center: 8.5 um ✓
    - Confidence: HIGH
    ```

### What You Should Output

Print a detailed report like this:

```
=== DISCOVERY PHASE REPORT ===

IMAGE CHARACTERISTICS:
- Image dimensions: 512 x 512
- Pixel value range: [min, max]
- Background intensity: X ± Y
- Peak intensity (brightest): Z

OBJECT DETECTION RESULTS:
- Method tried: Blob detection (blob_log)
- Parameters used: min_sigma=3, max_sigma=15, threshold=0.1
- Objects detected: 5
- Object positions: (y1, x1), (y2, x2), ...
- Object sizes (pixel diameter): size1, size2, ...
- Confidence scores: conf1, conf2, ...

CLOSEST OBJECT TO CENTER:
- Position: (y, x) pixels
- Distance from center: D um
- Size: S pixels
- Confidence: C
- Assessment: "Object is within 10um, tracking is feasible" / "WARNING: Object is far from center"

DETECTION METHOD ASSESSMENT:
- Simple thresholding: [works/doesn't work because...]
- Blob detection: [works/doesn't work because...]
- Recommendation: "Use blob detection because..."

SCREENSHOT ABOVE SHOWS DETECTION RESULTS
Please review and provide feedback:
1. "looks good, proceed" → I'll start tracking
2. "adjust min_sigma to X" → I'll test new parameters
3. "too many false positives" → I'll try different method
4. Any other specific feedback

NEXT STEPS:
Awaiting user feedback on:
1. Does the detection look correct?
2. Are parameters appropriate?
3. Should I adjust detection method?
4. Any concerns before proceeding to tracking?
```

### Do NOT Yet Write

❌ Do NOT write the full tracking loop
❌ Do NOT commit to specific parameters
❌ Just analyze and report

---

## PHASE 2: FEEDBACK PHASE (User Reviews)

You wait for user feedback before proceeding.

### What User Will Provide

User will review your LEARN phase report and tell you:
- "Detection looks good, proceed with tracking"
- "Try adjusting threshold to X"
- "Try different detection method"
- "Adjust blob parameters to min_sigma=Y, max_sigma=Z"
- "Object is too far from center, abort"

### How to Handle Feedback

Once user provides feedback:

1. **If user says "detection looks good":**
   - Proceed directly to PHASE 3
   - Use parameters you discovered

2. **If user suggests adjustments:**
   - Re-test with adjusted parameters
   - Report results
   - Wait for final approval

3. **If user says "abort":**
   - Stop, explain why you think tracking won't work
   - Propose alternative (different region, different settings)

---

## PHASE 3: EXPERIMENT PHASE (Execute Protocol)

Only after user approval, execute the final tracking code.

### Structure

```python
# Only execute after user says: "Detection looks good, proceed"

# Use parameters discovered in PHASE 1
min_sigma = 3  # or whatever user approved
max_sigma = 15
threshold = 0.1

# Run full tracking protocol
# (same as before: detect → select closest → track for 1 minute)

print("=== TRACKING PROTOCOL ===")
print(f"Using detection parameters: min_sigma={min_sigma}, max_sigma={max_sigma}")
print("Starting 1-minute tracking...")

# ... full tracking code ...

print(f"Tracking complete! {len(images)} images captured")
```

### What to Return

- List of all tracked images
- Final summary statistics
- Trajectory of tracked object

---

## Summary of Your Job (Agent's Responsibilities)

| Phase | Agent Does | Agent Outputs | Status |
|-------|----------|--------------|--------|
| LEARN | Capture images, test methods | Detailed report, recommendations | Waits for feedback |
| FEEDBACK | Adjust based on user input, re-test if needed | Updated report if adjusted | Waits for approval |
| EXPERIMENT | Execute final tracking protocol | Images list + statistics | Done |

---

## Key Instructions

### Do These Things

✅ Be thorough in LEARN phase
✅ Print detailed findings
✅ Try multiple detection approaches
✅ Give clear recommendations
✅ Wait for user feedback before EXPERIMENT
✅ Use approved parameters in final tracking
✅ Print progress during tracking

### Don't Do These Things

❌ Skip discovery, jump to tracking
❌ Make assumptions about object sizes/brightness
❌ Hide your reasoning (explain what you found and why)
❌ Change parameters without user approval
❌ Use approximations instead of measurements
❌ Write tracking code until user says "approved"

---

## Execution Guidelines

### Execution Mode

- **LEARN Phase:** Use `execution_mode="buffered"`
  (Analysis code, no loops, batch operations)

- **EXPERIMENT Phase:** Use `execution_mode="live"`
  (Tracking loop requires real-time I/O)

### During LEARN Phase

Capture multiple images to test robustness:
```python
# Capture 3-5 images from different positions
for attempt in range(3):
    mmc.snapImage()
    img = mmc.getImage()
    
    # Analyze each
    objects = detect_all_objects(img)
    print(f"Attempt {attempt}: Found {len(objects)} objects")
```

This tests: Does detection work reliably or only sometimes?

### Expected Duration

- LEARN Phase: 2-3 minutes (hardware execution)
- FEEDBACK Phase: User review time (not your code)
- EXPERIMENT Phase: 1 minute (actual tracking)

Total: ~5-10 minutes for full workflow

---

## When to Ask for Help

If during LEARN phase you find:
- No objects detected anywhere → something is wrong, ask user
- Thousands of false detections → parameters too sensitive, ask user
- Objects extremely variable in size → may need adaptive approach, ask user
- Object never appears near center → may be impossible, ask user

Be transparent about uncertainty!
