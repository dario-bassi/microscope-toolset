"""Cell viability analysis.

Quantifies live/dead cell ratios from fluorescence markers,
trypan blue exclusion, and morphological criteria.

Functions:
    live_dead_count     -- Count live and dead cells from dual-stain images
    viability_index     -- Compute viability percentage from counts
    trypan_blue_count   -- Count viable/non-viable cells in trypan blue
    morphology_viability -- Assess viability from cell morphology
    viability_timecourse -- Track viability over multiple timepoints
    pixel_viability     -- Pixel-based live/dead classification from dual channels
    volume_corrected_viability -- Correct 2D slice dead-fraction for 3D spheroid
"""

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology


def live_dead_count(
    live_channel, dead_channel, min_area=20, live_threshold=None, dead_threshold=None
):
    """Count live and dead cells from dual fluorescence channels.

    Typical stains: Calcein-AM (live, green) / Ethidium homodimer (dead, red)
    or similar live/dead assay pairs.

    Args:
        live_channel: 2D array, fluorescence image for live marker.
        dead_channel: 2D array, fluorescence image for dead marker.
        min_area: int, minimum cell area in pixels.
        live_threshold: float or None, threshold for live channel.
            If None, uses Otsu.
        dead_threshold: float or None, threshold for dead channel.
            If None, uses Otsu.

    Returns:
        dict with:
            n_live: int, number of live cells.
            n_dead: int, number of dead cells.
            n_total: int, total cells detected.
            viability: float, fraction alive (0-1).
            live_mask: 2D bool array.
            dead_mask: 2D bool array.
    """
    live_channel = np.asarray(live_channel, dtype=float)
    dead_channel = np.asarray(dead_channel, dtype=float)

    # Threshold each channel
    if live_threshold is None:
        live_threshold = filters.threshold_otsu(live_channel)
    if dead_threshold is None:
        dead_threshold = filters.threshold_otsu(dead_channel)

    live_mask = live_channel > live_threshold
    dead_mask = dead_channel > dead_threshold

    # Remove small objects
    if live_mask.any():
        live_mask = morphology.remove_small_objects(live_mask, max_size=min_area - 1)
    if dead_mask.any():
        dead_mask = morphology.remove_small_objects(dead_mask, max_size=min_area - 1)

    # Remove overlap: if both channels positive, classify by stronger signal
    overlap = live_mask & dead_mask
    if overlap.any():
        # Normalize each channel to 0-1 range
        live_norm = live_channel / max(live_channel.max(), 1)
        dead_norm = dead_channel / max(dead_channel.max(), 1)
        # Where overlap exists, assign to stronger channel
        live_wins = live_norm > dead_norm
        live_mask[overlap] = live_wins[overlap]
        dead_mask[overlap] = ~live_wins[overlap]

    # Count connected components
    live_labeled = measure.label(live_mask)
    dead_labeled = measure.label(dead_mask)
    n_live = live_labeled.max()
    n_dead = dead_labeled.max()
    n_total = n_live + n_dead

    viability = n_live / n_total if n_total > 0 else 0.0

    return {
        "n_live": n_live,
        "n_dead": n_dead,
        "n_total": n_total,
        "viability": round(viability, 4),
        "live_mask": live_mask,
        "dead_mask": dead_mask,
    }


def viability_index(n_live, n_dead, method="fraction"):
    """Compute viability index from cell counts.

    Args:
        n_live: int, number of live cells.
        n_dead: int, number of dead cells.
        method: str, calculation method:
            'fraction' — n_live / (n_live + n_dead)
            'ratio' — n_live / max(n_dead, 1)
            'percent' — 100 * n_live / (n_live + n_dead)

    Returns:
        dict with:
            viability: float, computed viability metric.
            n_live: int.
            n_dead: int.
            n_total: int.
            method: str.
    """
    n_total = n_live + n_dead

    if method == "fraction":
        viability = n_live / n_total if n_total > 0 else 0.0
    elif method == "ratio":
        viability = n_live / max(n_dead, 1)
    elif method == "percent":
        viability = 100 * n_live / n_total if n_total > 0 else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "viability": round(float(viability), 4),
        "n_live": n_live,
        "n_dead": n_dead,
        "n_total": n_total,
        "method": method,
    }


def trypan_blue_count(image, cell_min_area=50, cell_max_area=5000, blue_threshold=None):
    """Count viable and non-viable cells in trypan blue exclusion.

    Dead cells take up trypan blue (appear dark blue), live cells
    exclude the dye (appear bright/refractile).

    Args:
        image: 2D array, brightfield image of trypan blue stained cells.
        cell_min_area: int, minimum cell area in pixels.
        cell_max_area: int, maximum cell area in pixels.
        blue_threshold: float or None, threshold for blue staining.
            Below this = dead (blue). If None, uses median - 1 std.

    Returns:
        dict with:
            n_viable: int.
            n_nonviable: int.
            n_total: int.
            viability: float (0-1).
            cell_props: list of regionprops with 'viable' attribute added.
    """
    image = np.asarray(image, dtype=float)

    # Detect all cells (both live and dead appear as objects)
    bg = np.median(image)
    cell_mask = np.abs(image - bg) > max(np.std(image) * 0.5, 5)
    cell_mask = ndimage.binary_fill_holes(cell_mask)
    if cell_mask.any():
        cell_mask = morphology.remove_small_objects(cell_mask, max_size=cell_min_area - 1)

    labeled = measure.label(cell_mask)
    props = measure.regionprops(labeled, intensity_image=image)

    # Filter by size
    valid_props = [p for p in props if cell_min_area <= p.area <= cell_max_area]

    if not valid_props:
        return {
            "n_viable": 0,
            "n_nonviable": 0,
            "n_total": 0,
            "viability": 0.0,
            "cell_props": [],
        }

    # Classify each cell
    intensities = np.array([p.intensity_mean for p in valid_props])
    if blue_threshold is None:
        blue_threshold = np.median(intensities) - 0.5 * np.std(intensities)

    n_viable = 0
    n_nonviable = 0
    for p in valid_props:
        if p.intensity_mean < blue_threshold:
            p.viable = False
            n_nonviable += 1
        else:
            p.viable = True
            n_viable += 1

    n_total = n_viable + n_nonviable
    viability = n_viable / n_total if n_total > 0 else 0.0

    return {
        "n_viable": n_viable,
        "n_nonviable": n_nonviable,
        "n_total": n_total,
        "viability": round(viability, 4),
        "cell_props": valid_props,
    }


def morphology_viability(props, circularity_thresh=0.6, area_cv_thresh=0.8):
    """Assess viability from cell morphology (no staining required).

    Dead/dying cells become irregular, fragmented, and variable in size.
    Healthy cells are round and uniform.

    Args:
        props: list of regionprops from segmented cells.
        circularity_thresh: float, cells below this are considered
            abnormal. Circularity = 4*pi*area / perimeter^2.
        area_cv_thresh: float, if population CV > this, suggests
            significant cell death/damage.

    Returns:
        dict with:
            n_normal: int, cells with normal morphology.
            n_abnormal: int, cells with abnormal morphology.
            n_total: int.
            morphology_score: float, fraction normal (0-1).
            area_cv: float, coefficient of variation of cell areas.
            mean_circularity: float.
    """
    if not props:
        return {
            "n_normal": 0,
            "n_abnormal": 0,
            "n_total": 0,
            "morphology_score": 0.0,
            "area_cv": 0.0,
            "mean_circularity": 0.0,
        }

    circularities = []
    areas = []
    n_normal = 0
    n_abnormal = 0

    for p in props:
        area = p.area
        perim = p.perimeter
        circ = 4 * np.pi * area / (perim**2) if perim > 0 else 0
        circularities.append(circ)
        areas.append(area)

        if circ >= circularity_thresh:
            n_normal += 1
        else:
            n_abnormal += 1

    areas = np.array(areas)
    area_cv = float(areas.std() / areas.mean()) if areas.mean() > 0 else 0

    n_total = n_normal + n_abnormal
    score = n_normal / n_total if n_total > 0 else 0.0

    return {
        "n_normal": n_normal,
        "n_abnormal": n_abnormal,
        "n_total": n_total,
        "morphology_score": round(score, 4),
        "area_cv": round(area_cv, 4),
        "mean_circularity": round(float(np.mean(circularities)), 4),
    }


def pixel_viability(live_channel, dead_channel, background_threshold=None):
    """Pixel-based live/dead classification from dual fluorescence channels.

    Classifies each pixel as background, live, or dead based on relative
    channel intensities. Useful for spatial viability mapping in organoids
    or spheroids where single-cell segmentation is impractical.

    Args:
        live_channel: 2D array, fluorescence image for live marker
            (e.g., Calcein-AM green channel).
        dead_channel: 2D array, fluorescence image for dead marker
            (e.g., Ethidium homodimer red channel).
        background_threshold: float or None. Pixels below this in BOTH
            channels are considered background. If None, uses the lower
            Otsu threshold of the two channels.

    Returns:
        dict with:
            pixel_map: 2D uint8 array (0=background, 1=live, 2=dead).
            fraction_live: float, live pixels / foreground pixels.
            fraction_dead: float, dead pixels / foreground pixels.
            fraction_background: float, background pixels / total.
            pixel_viability: float, fraction_live (alias).
    """
    live = np.asarray(live_channel, dtype=float)
    dead = np.asarray(dead_channel, dtype=float)

    if background_threshold is None:
        t_live = filters.threshold_otsu(live) if live.max() > 0 else 0
        t_dead = filters.threshold_otsu(dead) if dead.max() > 0 else 0
        background_threshold = min(t_live, t_dead)

    foreground = (live > background_threshold) | (dead > background_threshold)

    # Among foreground pixels: classify by dominant channel
    live_norm = live / max(live.max(), 1e-9)
    dead_norm = dead / max(dead.max(), 1e-9)

    pixel_map = np.zeros(live.shape, dtype=np.uint8)
    fg_live = foreground & (live_norm >= dead_norm)
    fg_dead = foreground & (dead_norm > live_norm)
    pixel_map[fg_live] = 1
    pixel_map[fg_dead] = 2

    n_fg = int(foreground.sum())
    n_live = int(fg_live.sum())
    n_dead = int(fg_dead.sum())
    n_total = live.size

    fraction_live = n_live / n_fg if n_fg > 0 else 0.0
    fraction_dead = n_dead / n_fg if n_fg > 0 else 0.0
    fraction_background = 1.0 - n_fg / n_total

    return {
        "pixel_map": pixel_map,
        "fraction_live": round(fraction_live, 4),
        "fraction_dead": round(fraction_dead, 4),
        "fraction_background": round(fraction_background, 4),
        "pixel_viability": round(fraction_live, 4),
    }


def volume_corrected_viability(dead_fraction_2d, geometry="spherical"):
    """Correct 2D slice dead-fraction to 3D volumetric dead fraction.

    When imaging a spheroid/organoid at its equatorial plane, the 2D
    area fraction of dead (necrotic core) underestimates or differently
    represents the true 3D volume fraction.

    Derivation for spherical geometry (equatorial slice):
        f_2d = (r_core / r_spheroid)^2  (area fraction in 2D slice)
        f_3d = (r_core / r_spheroid)^3  (volume fraction in 3D)
        → f_3d = f_2d^(3/2)

    Args:
        dead_fraction_2d: float or array-like, dead area fraction from
            2D equatorial slice (0-1). Values are clipped to [0, 1].
        geometry: str, correction geometry:
            'spherical' — assumes spherical organoid, equatorial imaging.
                f_3d = f_2d^1.5
            'cylindrical' — assumes cylindrical shape (cross-section slice).
                f_3d ≈ f_2d (no radial correction needed)
            'flat' — no correction; f_3d = f_2d (thin monolayer).

    Returns:
        dict with:
            dead_fraction_2d: float, input value.
            dead_fraction_3d: float, volumetric dead fraction.
            live_fraction_3d: float, 1 - dead_fraction_3d.
            geometry: str.
    """
    f2d = np.clip(np.asarray(dead_fraction_2d, dtype=float), 0.0, 1.0)

    if geometry == "spherical":
        f3d = f2d**1.5
    elif geometry in ("cylindrical", "flat"):
        f3d = f2d
    else:
        raise ValueError(
            f"Unknown geometry: {geometry!r}. " "Use 'spherical', 'cylindrical', or 'flat'."
        )

    scalar = f2d.ndim == 0
    f2d_out = float(f2d) if scalar else f2d.tolist()
    f3d_out = float(f3d) if scalar else f3d.tolist()
    live_out = float(1 - f3d) if scalar else (1 - f3d).tolist()

    return {
        "dead_fraction_2d": f2d_out,
        "dead_fraction_3d": round(f3d_out, 6) if scalar else f3d_out,
        "live_fraction_3d": round(live_out, 6) if scalar else live_out,
        "geometry": geometry,
    }


def viability_timecourse(timepoints, viabilities):
    """Analyze viability over time.

    Computes rate of change, time to half-death, and classifies
    the response pattern.

    Args:
        timepoints: array-like, time values.
        viabilities: array-like, viability fraction at each timepoint.

    Returns:
        dict with:
            initial_viability: float, first timepoint.
            final_viability: float, last timepoint.
            change: float, final - initial.
            rate: float, change per unit time.
            half_death_time: float or None, time to reach 50%
                of initial viability (if declining).
            pattern: str, 'stable', 'declining', 'recovering',
                or 'variable'.
    """
    t = np.asarray(timepoints, dtype=float)
    v = np.asarray(viabilities, dtype=float)

    if len(v) < 2:
        return {
            "initial_viability": float(v[0]) if len(v) > 0 else 0,
            "final_viability": float(v[-1]) if len(v) > 0 else 0,
            "change": 0.0,
            "rate": 0.0,
            "half_death_time": None,
            "pattern": "stable",
        }

    initial = float(v[0])
    final = float(v[-1])
    change = final - initial
    dt = t[-1] - t[0]
    rate = change / dt if dt > 0 else 0.0

    # Half-death time: when viability drops to 50% of initial
    half_death_time = None
    if initial > 0 and change < 0:
        target = initial * 0.5
        below = np.where(v <= target)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                # Linear interpolation
                t0, t1 = t[idx - 1], t[idx]
                v0, v1 = v[idx - 1], v[idx]
                if v0 != v1:
                    half_death_time = float(t0 + (target - v0) / (v1 - v0) * (t1 - t0))
                else:
                    half_death_time = float(t0)
            else:
                half_death_time = float(t[0])

    # Pattern classification
    if abs(change) < 0.05:
        # Check for variability
        if np.std(v) > 0.1:
            pattern = "variable"
        else:
            pattern = "stable"
    elif change < -0.1:
        pattern = "declining"
    elif change > 0.1:
        pattern = "recovering"
    else:
        pattern = "stable"

    return {
        "initial_viability": round(initial, 4),
        "final_viability": round(final, 4),
        "change": round(change, 4),
        "rate": round(rate, 6),
        "half_death_time": round(half_death_time, 4) if half_death_time is not None else None,
        "pattern": pattern,
    }
