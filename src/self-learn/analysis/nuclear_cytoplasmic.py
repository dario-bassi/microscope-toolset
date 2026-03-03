"""Nuclear-cytoplasmic ratio analysis.

Quantifies the distribution of fluorescent signal between nucleus
and cytoplasm. Useful for translocation assays, stress response,
cell cycle analysis, and nuclear transport studies.

Functions:
    segment_cytoplasm       -- Derive cytoplasm mask from cell and nuclear masks
    compute_nc_ratio        -- Nuclear / cytoplasmic intensity ratio per cell
    classify_localization   -- Classify as nuclear, cytoplasmic, or uniform
    translocation_timecourse -- Track N:C ratio over time for translocation kinetics
    population_nc_stats     -- Population-level N:C ratio statistics
"""

import numpy as np
from scipy import ndimage


def segment_cytoplasm(cell_mask, nuclear_mask, min_cytoplasm_area=10):
    """Derive cytoplasm mask from whole-cell and nuclear masks.

    Cytoplasm = cell mask minus nuclear mask. Labels are matched
    by overlap: each nuclear region is assigned to the cell label
    that contains it.

    Args:
        cell_mask: 2D labeled array (0=background, 1..N=cells).
        nuclear_mask: 2D labeled array (0=background, 1..M=nuclei).
        min_cytoplasm_area: int, minimum cytoplasm pixels to keep.

    Returns:
        dict with:
            cytoplasm_mask: 2D labeled array, matched to cell labels.
            matched_pairs: list of (cell_label, nucleus_label) tuples.
            unmatched_cells: list of cell labels without nuclei.
            unmatched_nuclei: list of nucleus labels without cells.
    """
    cell_mask = np.asarray(cell_mask)
    nuclear_mask = np.asarray(nuclear_mask)

    cell_labels = set(np.unique(cell_mask)) - {0}
    nuc_labels = set(np.unique(nuclear_mask)) - {0}

    matched = []
    nuc_to_cell = {}

    # Match each nucleus to its enclosing cell
    for nuc_id in nuc_labels:
        nuc_pixels = nuclear_mask == nuc_id
        overlapping = cell_mask[nuc_pixels]
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) > 0:
            best_cell = int(np.bincount(overlapping).argmax())
            matched.append((best_cell, nuc_id))
            nuc_to_cell[nuc_id] = best_cell

    matched_cells = {pair[0] for pair in matched}
    matched_nucs = {pair[1] for pair in matched}

    # Build cytoplasm mask
    cyto = np.zeros_like(cell_mask)
    for cell_id, nuc_id in matched:
        cell_region = cell_mask == cell_id
        nuc_region = nuclear_mask == nuc_id
        cyto_region = cell_region & ~nuc_region
        if cyto_region.sum() >= min_cytoplasm_area:
            cyto[cyto_region] = cell_id

    return {
        'cytoplasm_mask': cyto,
        'matched_pairs': matched,
        'unmatched_cells': sorted(cell_labels - matched_cells),
        'unmatched_nuclei': sorted(nuc_labels - matched_nucs),
    }


def compute_nc_ratio(image, nuclear_mask, cytoplasm_mask, background=None):
    """Compute nuclear-to-cytoplasmic intensity ratio per cell.

    N:C ratio = mean(nuclear intensity) / mean(cytoplasmic intensity)
    for each matched cell-nucleus pair.

    Args:
        image: 2D array, fluorescence image.
        nuclear_mask: 2D labeled array.
        cytoplasm_mask: 2D labeled array (from segment_cytoplasm).
        background: float or None, background value to subtract.

    Returns:
        dict with:
            ratios: dict mapping cell_label → N:C ratio.
            nuclear_intensities: dict mapping cell_label → mean nuclear intensity.
            cytoplasmic_intensities: dict mapping cell_label → mean cytoplasmic intensity.
            mean_ratio: float, population mean N:C ratio.
            median_ratio: float, population median N:C ratio.
    """
    image = np.asarray(image, dtype=float)
    nuclear_mask = np.asarray(nuclear_mask)
    cytoplasm_mask = np.asarray(cytoplasm_mask)

    if background is not None:
        image = image - background

    nuc_labels = set(np.unique(nuclear_mask)) - {0}
    cyto_labels = set(np.unique(cytoplasm_mask)) - {0}
    common = nuc_labels & cyto_labels

    ratios = {}
    nuc_int = {}
    cyto_int = {}

    for label in sorted(common):
        nuc_region = nuclear_mask == label
        cyto_region = cytoplasm_mask == label

        n_mean = float(image[nuc_region].mean())
        c_mean = float(image[cyto_region].mean()) if cyto_region.sum() > 0 else 0.0

        nuc_int[label] = round(n_mean, 4)
        cyto_int[label] = round(c_mean, 4)

        if c_mean > 0:
            ratios[label] = round(n_mean / c_mean, 4)
        else:
            ratios[label] = float('inf') if n_mean > 0 else 1.0

    ratio_vals = [v for v in ratios.values() if np.isfinite(v)]
    mean_r = float(np.mean(ratio_vals)) if ratio_vals else 0.0
    median_r = float(np.median(ratio_vals)) if ratio_vals else 0.0

    return {
        'ratios': ratios,
        'nuclear_intensities': nuc_int,
        'cytoplasmic_intensities': cyto_int,
        'mean_ratio': round(mean_r, 4),
        'median_ratio': round(median_r, 4),
    }


def classify_localization(ratios, nuclear_threshold=2.0, cytoplasmic_threshold=0.5):
    """Classify signal localization per cell based on N:C ratio.

    Args:
        ratios: dict mapping cell_label → N:C ratio, or list of ratios.
        nuclear_threshold: float, N:C ratio above which signal is nuclear.
        cytoplasmic_threshold: float, N:C ratio below which signal is cytoplasmic.

    Returns:
        dict with:
            classifications: dict or list mapping to 'nuclear'/'cytoplasmic'/'uniform'.
            counts: dict of category counts.
            fractions: dict of category fractions.
    """
    if isinstance(ratios, dict):
        items = ratios.items()
        as_dict = True
    else:
        items = enumerate(ratios)
        as_dict = False

    classifications = {}
    for key, r in items:
        if not np.isfinite(r):
            classifications[key] = 'nuclear'
        elif r >= nuclear_threshold:
            classifications[key] = 'nuclear'
        elif r <= cytoplasmic_threshold:
            classifications[key] = 'cytoplasmic'
        else:
            classifications[key] = 'uniform'

    counts = {'nuclear': 0, 'cytoplasmic': 0, 'uniform': 0}
    for cls in classifications.values():
        counts[cls] += 1

    total = max(len(classifications), 1)
    fractions = {k: round(v / total, 4) for k, v in counts.items()}

    result_cls = classifications if as_dict else list(classifications.values())

    return {
        'classifications': result_cls,
        'counts': counts,
        'fractions': fractions,
    }


def translocation_timecourse(images, nuclear_mask, cytoplasm_mask,
                              timepoints=None, background=None):
    """Track N:C ratio over time for translocation kinetics.

    Computes mean population N:C ratio at each timepoint.

    Args:
        images: 3D array (T, H, W) or list of 2D images.
        nuclear_mask: 2D labeled array (static segmentation).
        cytoplasm_mask: 2D labeled array (static segmentation).
        timepoints: 1D array of time values, or None (uses frame indices).
        background: float or None.

    Returns:
        dict with:
            timepoints: 1D array.
            mean_ratios: 1D array, population mean N:C ratio per frame.
            median_ratios: 1D array, population median N:C ratio per frame.
            fold_change: float, final ratio / initial ratio.
            max_ratio: float, peak N:C ratio.
            time_to_peak: float, time at which max ratio occurs.
    """
    images = np.asarray(images, dtype=float)
    n_frames = images.shape[0]

    if timepoints is None:
        timepoints = np.arange(n_frames, dtype=float)
    else:
        timepoints = np.asarray(timepoints, dtype=float)

    mean_ratios = np.zeros(n_frames)
    median_ratios = np.zeros(n_frames)

    for t in range(n_frames):
        result = compute_nc_ratio(images[t], nuclear_mask, cytoplasm_mask,
                                   background=background)
        mean_ratios[t] = result['mean_ratio']
        median_ratios[t] = result['median_ratio']

    initial = mean_ratios[0] if mean_ratios[0] > 0 else 1.0
    fold = float(mean_ratios[-1] / initial)
    peak_idx = int(np.argmax(mean_ratios))

    return {
        'timepoints': timepoints,
        'mean_ratios': mean_ratios,
        'median_ratios': median_ratios,
        'fold_change': round(fold, 4),
        'max_ratio': round(float(mean_ratios[peak_idx]), 4),
        'time_to_peak': float(timepoints[peak_idx]),
    }


def population_nc_stats(ratios, bins=None):
    """Compute population-level N:C ratio statistics.

    Args:
        ratios: list or array of N:C ratio values.
        bins: int or None, number of histogram bins.

    Returns:
        dict with:
            mean: float.
            median: float.
            std: float.
            cv: float, coefficient of variation.
            q25: float, 25th percentile.
            q75: float, 75th percentile.
            iqr: float, interquartile range.
            histogram: tuple (counts, bin_edges) if bins given.
            n: int, number of cells.
    """
    ratios = np.asarray([r for r in ratios if np.isfinite(r)], dtype=float)

    if len(ratios) == 0:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0, 'cv': 0.0,
            'q25': 0.0, 'q75': 0.0, 'iqr': 0.0, 'n': 0,
        }

    mean_val = float(np.mean(ratios))
    std_val = float(np.std(ratios))
    q25 = float(np.percentile(ratios, 25))
    q75 = float(np.percentile(ratios, 75))

    result = {
        'mean': round(mean_val, 4),
        'median': round(float(np.median(ratios)), 4),
        'std': round(std_val, 4),
        'cv': round(std_val / mean_val, 4) if mean_val > 0 else 0.0,
        'q25': round(q25, 4),
        'q75': round(q75, 4),
        'iqr': round(q75 - q25, 4),
        'n': len(ratios),
    }

    if bins is not None:
        hist, edges = np.histogram(ratios, bins=bins)
        result['histogram'] = (hist, edges)

    return result
