# Failure Analysis: Autofocus Range Selection & PFS Calibration

## Challenge Type
Software autofocus on a real Nikon Ti-E with unknown sample Z position.

## What Went Wrong

### Error 1: Search range too narrow
- **Error**: Used default autofocus with +/- 200um range from current Z (~3369um). Focus metric curve was flat (<6% variation across all positions).
- **Root cause**: True focal plane was at Z=3256, about 140um outside the search range.
- **Diagnostic**: A flat focus metric curve (< 10% variation) means the focal plane is entirely outside the search range. A good autofocus scan should show a clear peak with >2x variation.
- **Fix**: Widened coarse scan to cover 3200-3600um. Found clear peak at Z=3256.

### Error 2: TIZDrive timeout on large Z moves
- **Error**: `RuntimeError: Wait for device "TIZDrive" timed out after 10000ms` when scanning +/- 200um.
- **Root cause**: Default Core timeout (10s) is too short for the Nikon TIZDrive when making large Z moves.
- **Fix**: `mmc.setProperty("Core", "TimeoutMs", 30000)` before autofocus.

### Error 3: PFS offset not calibrated
- **Error**: Enabled PFS (Perfect Focus System) with default offset 144.5 → locked at Z=3114, but true focus was at Z=3145 (31um off).
- **Root cause**: PFS offset must be calibrated per sample/objective. The offset determines which Z plane PFS locks to.
- **Fix**: Scanned PFS offsets (134.5 to 199.5 in steps of 5) while measuring Tenengrad focus metric on captured images. Best offset = 169.5.

## Key Diagnostic Rules

1. **Flat focus curve = wrong range**: If all Z positions give similar metric values, the focus plane is far from the search window. Widen the range or start from a different Z.
2. **Good focus curve**: Should show a single clear peak. Metric at peak should be >2x the metric at the edges of the range.
3. **PFS is not plug-and-play**: After enabling PFS, always verify focus quality by snapping an image and computing a sharpness metric. Calibrate offset if needed.

## Correct Approach

1. Increase Core timeout to 30s before any large Z moves
2. Start with a wide coarse scan (e.g., +/- 300um, 20um steps)
3. Check if focus curve has a clear peak. If flat, widen further
4. Once coarse peak found, do fine scan (+/- 15um, 1um steps)
5. If PFS available: enable it, then calibrate offset by scanning offsets and measuring image sharpness
6. PFS maintains focus during XY moves (essential for tile scans)

## Hardware Context
- Nikon Ti-E, TIZDrive Z-stage
- PFS (TIPFSStatus / TIPFSOffset devices)
- Needed 30s timeout for Z moves spanning >200um
