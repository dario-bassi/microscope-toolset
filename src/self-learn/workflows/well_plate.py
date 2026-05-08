"""Multi-well plate imaging workflow.

Systematic microscopy imaging of multi-well plates with per-well
analysis and cross-well comparison. Handles plate geometry, well
naming, MDA-based acquisition, and statistical comparison.

Complements plate_reader.py (scalar data) with imaging-specific
functionality: stage positions from well layouts, multi-channel
acquisition per well, per-well cell analysis, and condition
comparison with proper statistics.

Functions:
    well_layout        -- Plate geometry and well center positions
    well_name          -- Convert (row, col) to well name (e.g. "A1")
    parse_well_name    -- Convert well name to (row, col)
    well_scan_events   -- MDA generator for systematic well imaging
    per_well_summary   -- Aggregate measurements per well
    compare_wells      -- Statistical comparison across wells/conditions
    assign_conditions  -- Map well names to experimental conditions
"""

import numpy as np
from useq import MDAEvent


# Standard plate formats: (n_rows, n_cols, well_spacing_mm, well_diameter_mm)
PLATE_FORMATS = {
    6:   (2, 3, 39.12, 34.8),
    12:  (3, 4, 26.01, 22.1),
    24:  (4, 6, 19.30, 15.6),
    48:  (6, 8, 13.00, 11.0),
    96:  (8, 12, 9.00, 6.35),
    384: (16, 24, 4.50, 3.30),
}


def well_layout(n_wells=96, spacing_um=None, origin_um=None,
                selected_wells=None):
    """Generate plate layout with well center positions.

    Args:
        n_wells: Plate format (6, 12, 24, 48, 96, 384).
        spacing_um: Well-to-well spacing in microns. If None, uses
            standard spacing for the plate format (converted from mm).
        origin_um: (x, y) stage position of well A1 center in microns.
            Defaults to (0, 0).
        selected_wells: Optional list of well names (e.g. ["A1", "B3"])
            to include. If None, includes all wells.

    Returns:
        dict with:
            wells: dict mapping well_name -> {'row': int, 'col': int,
                'x': float, 'y': float} with stage positions in microns.
            n_rows: int, number of rows.
            n_cols: int, number of columns.
            spacing_um: float, well spacing.
            plate_format: int, number of wells.
            well_names: list of str, ordered well names.
    """
    if n_wells not in PLATE_FORMATS:
        raise ValueError(
            f"Unknown plate format: {n_wells}. "
            f"Supported: {sorted(PLATE_FORMATS.keys())}"
        )

    n_rows, n_cols, spacing_mm, _ = PLATE_FORMATS[n_wells]

    if spacing_um is None:
        spacing_um = spacing_mm * 1000  # mm -> um

    if origin_um is None:
        origin_um = (0.0, 0.0)

    ox, oy = float(origin_um[0]), float(origin_um[1])

    wells = {}
    well_names = []

    for row in range(n_rows):
        for col in range(n_cols):
            name = well_name(row, col)

            if selected_wells is not None:
                if name not in selected_wells:
                    continue

            x = ox + col * spacing_um
            y = oy + row * spacing_um

            wells[name] = {
                'row': row,
                'col': col,
                'x': float(x),
                'y': float(y),
            }
            well_names.append(name)

    return {
        'wells': wells,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'spacing_um': float(spacing_um),
        'plate_format': n_wells,
        'well_names': well_names,
    }


def well_name(row, col):
    """Convert (row, col) to well name.

    Args:
        row: 0-indexed row (0=A, 1=B, ...).
        col: 0-indexed column (0=1, 1=2, ...).

    Returns:
        str, e.g. "A1", "B12", "H8".
    """
    letter = chr(ord('A') + row)
    return f"{letter}{col + 1}"


def parse_well_name(name):
    """Convert well name to (row, col).

    Args:
        name: Well name like "A1", "B12", "H8".

    Returns:
        tuple (row, col), 0-indexed.
    """
    name = name.strip().upper()
    letter = name[0]
    number = int(name[1:])
    row = ord(letter) - ord('A')
    col = number - 1
    return (row, col)


def well_scan_events(layout, channels=None, exposure=50.0,
                     n_sites_per_well=1, site_offsets=None,
                     channel_group=None, z_positions=None):
    """Generate MDA events for systematic well plate imaging.

    Visits each well in the layout, optionally imaging multiple sites
    per well and multiple channels per site.

    Args:
        layout: dict from well_layout() with 'wells' key.
        channels: list of channel config names (e.g. ["DAPI", "GFP"]).
            If None, acquires without channel config.
        exposure: Exposure time in ms (float or dict mapping channel->ms).
        n_sites_per_well: Number of imaging sites per well.
            1 = center only. >1 = grid within well.
        site_offsets: Optional list of (dx, dy) offsets from well center
            for each site. If None and n_sites_per_well > 1, generates
            a grid of offsets.
        channel_group: Config group name (e.g. "Channel", "Fake").
        z_positions: Optional list of Z positions for Z-stack per site.

    Returns:
        Generator of MDAEvent objects.
    """
    wells = layout if isinstance(layout, dict) and 'wells' in layout else layout
    if 'wells' in wells:
        well_dict = wells['wells']
        well_names = wells.get('well_names', sorted(well_dict.keys()))
    else:
        well_dict = wells
        well_names = sorted(wells.keys())

    # Generate site offsets if needed
    if site_offsets is None:
        if n_sites_per_well == 1:
            site_offsets = [(0.0, 0.0)]
        else:
            site_offsets = _grid_offsets(n_sites_per_well)
    else:
        site_offsets = [(float(dx), float(dy)) for dx, dy in site_offsets]

    ch_list = channels or [None]
    z_list = z_positions or [None]

    def _generator():
        for w_idx, wname in enumerate(well_names):
            info = well_dict[wname]
            wx, wy = info['x'], info['y']

            for s_idx, (dx, dy) in enumerate(site_offsets):
                sx = wx + dx
                sy = wy + dy

                for z_idx, z_pos in enumerate(z_list):
                    for c_idx, ch in enumerate(ch_list):
                        idx = {"p": w_idx * len(site_offsets) + s_idx}
                        if len(ch_list) > 1:
                            idx["c"] = c_idx
                        if len(z_list) > 1:
                            idx["z"] = z_idx

                        meta = {
                            "well_name": wname,
                            "well_index": w_idx,
                            "site_index": s_idx,
                        }

                        kwargs = {
                            "index": idx,
                            "exposure": _get_exposure(exposure, ch),
                            "x_pos": float(sx),
                            "y_pos": float(sy),
                            "metadata": meta,
                        }
                        if ch is not None:
                            ch_kwargs = {"config": ch}
                            if channel_group is not None:
                                ch_kwargs["group"] = channel_group
                            kwargs["channel"] = ch_kwargs
                        if z_pos is not None:
                            kwargs["z_pos"] = float(z_pos)

                        yield MDAEvent(**kwargs)

    return _generator


def per_well_summary(measurements, well_names=None):
    """Aggregate measurements per well.

    Takes a dict or list of per-well measurements and computes
    summary statistics (mean, std, count, etc.) for each well.

    Args:
        measurements: dict mapping well_name -> list of values,
            OR list of dicts with 'well' and 'value' keys.
        well_names: Optional list of well names to include.
            If None, includes all.

    Returns:
        dict mapping well_name -> {
            mean: float, std: float, median: float,
            count: int, values: list, sem: float, cv: float
        }
    """
    # Normalize input
    if isinstance(measurements, list):
        grouped = {}
        for item in measurements:
            wn = item['well']
            val = item['value']
            grouped.setdefault(wn, []).append(val)
        measurements = grouped

    result = {}
    for wn, values in measurements.items():
        if well_names is not None and wn not in well_names:
            continue

        vals = np.array(values, dtype=float)
        n = len(vals)

        if n == 0:
            result[wn] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'count': 0, 'values': [], 'sem': 0.0, 'cv': 0.0,
            }
            continue

        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem_v = std_v / np.sqrt(n) if n > 1 else 0.0
        cv_v = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else 0.0

        result[wn] = {
            'mean': round(mean_v, 4),
            'std': round(std_v, 4),
            'median': round(float(np.median(vals)), 4),
            'count': n,
            'values': vals.tolist(),
            'sem': round(sem_v, 4),
            'cv': round(cv_v, 4),
        }

    return result


def compare_wells(measurements, conditions, metric='mean',
                  test='auto'):
    """Statistical comparison of wells across conditions.

    Groups wells by condition, aggregates per-well metrics, and
    performs statistical tests between conditions.

    Args:
        measurements: dict mapping well_name -> list of values
            (from per_well_summary or raw data).
        conditions: dict mapping well_name -> condition_label.
        metric: How to summarize each well: 'mean', 'median', 'count'.
        test: Statistical test: 'auto' (picks based on n_groups),
            'ttest', 'mannwhitney', 'kruskal', 'anova'.

    Returns:
        dict with:
            condition_stats: dict mapping condition -> {mean, std, n, wells}.
            test_name: str, statistical test used.
            statistic: float, test statistic.
            p_value: float.
            significant: bool (p < 0.05).
            ranking: list of conditions sorted by metric (descending).
            pairwise: list of pairwise comparison dicts (if >2 groups).
    """
    from scipy import stats as sp_stats

    # Group wells by condition
    condition_wells = {}
    for wn, cond in conditions.items():
        condition_wells.setdefault(cond, []).append(wn)

    # Compute per-well metric
    condition_values = {}
    condition_stats = {}

    for cond, well_list in condition_wells.items():
        values = []
        for wn in well_list:
            if wn not in measurements:
                continue
            well_vals = measurements[wn]
            if isinstance(well_vals, dict):
                well_vals = well_vals.get('values', [])
            if not well_vals:
                continue

            arr = np.array(well_vals, dtype=float)
            if metric == 'mean':
                values.append(float(np.mean(arr)))
            elif metric == 'median':
                values.append(float(np.median(arr)))
            elif metric == 'count':
                values.append(float(len(arr)))

        condition_values[cond] = values
        n = len(values)
        condition_stats[cond] = {
            'mean': round(float(np.mean(values)), 4) if values else 0.0,
            'std': round(float(np.std(values, ddof=1)), 4) if n > 1 else 0.0,
            'n': n,
            'wells': well_list,
            'values': values,
        }

    # Statistical test
    groups = [np.array(v) for v in condition_values.values() if len(v) > 0]
    cond_names = [c for c, v in condition_values.items() if len(v) > 0]
    n_groups = len(groups)

    stat_val = 0.0
    p_val = 1.0
    test_name = 'none'

    if n_groups < 2:
        test_name = 'insufficient_groups'
    elif n_groups == 2:
        if test == 'auto':
            # Use t-test if enough samples and roughly normal
            if min(len(g) for g in groups) >= 3:
                test = 'ttest'
            else:
                test = 'mannwhitney'

        if test == 'ttest':
            stat_val, p_val = sp_stats.ttest_ind(groups[0], groups[1])
            test_name = 'welch_t_test'
        elif test == 'mannwhitney':
            if len(groups[0]) > 0 and len(groups[1]) > 0:
                stat_val, p_val = sp_stats.mannwhitneyu(
                    groups[0], groups[1], alternative='two-sided')
                test_name = 'mann_whitney_u'
        else:
            test_name = test
    else:
        # 3+ groups
        if test in ('auto', 'kruskal'):
            stat_val, p_val = sp_stats.kruskal(*groups)
            test_name = 'kruskal_wallis'
        elif test == 'anova':
            stat_val, p_val = sp_stats.f_oneway(*groups)
            test_name = 'one_way_anova'

    # Ranking
    ranking = sorted(condition_stats.keys(),
                     key=lambda c: condition_stats[c]['mean'],
                     reverse=True)

    # Pairwise comparisons (if >2 groups and significant)
    pairwise = []
    if n_groups > 2 and p_val < 0.05:
        for i in range(len(cond_names)):
            for j in range(i + 1, len(cond_names)):
                g1 = condition_values[cond_names[i]]
                g2 = condition_values[cond_names[j]]
                if len(g1) >= 2 and len(g2) >= 2:
                    s, p = sp_stats.mannwhitneyu(
                        g1, g2, alternative='two-sided')
                else:
                    s, p = 0.0, 1.0
                pairwise.append({
                    'pair': (cond_names[i], cond_names[j]),
                    'statistic': float(s),
                    'p_value': float(p),
                    'significant': p < 0.05,
                })

    return {
        'condition_stats': condition_stats,
        'test_name': test_name,
        'statistic': float(stat_val) if not np.isnan(stat_val) else 0.0,
        'p_value': float(p_val) if not np.isnan(p_val) else 1.0,
        'significant': bool(p_val < 0.05),
        'ranking': ranking,
        'pairwise': pairwise,
    }


def assign_conditions(well_names, condition_map):
    """Map well names to experimental conditions.

    Convenience function for creating condition assignments
    from various input formats.

    Args:
        well_names: list of well name strings.
        condition_map: dict mapping condition_label -> list of well names,
            OR dict mapping well_name -> condition_label.

    Returns:
        dict mapping well_name -> condition_label.
    """
    # Check if it's condition -> [wells] or well -> condition
    first_val = next(iter(condition_map.values()))
    if isinstance(first_val, (list, tuple)):
        # condition -> [wells]
        result = {}
        for cond, wells in condition_map.items():
            for w in wells:
                if w in well_names:
                    result[w] = cond
        return result
    else:
        # well -> condition (just filter to well_names)
        return {w: condition_map[w] for w in well_names
                if w in condition_map}


# --- Internal helpers ---

def _grid_offsets(n_sites, spacing=50.0):
    """Generate a grid of (dx, dy) offsets for multi-site imaging."""
    side = int(np.ceil(np.sqrt(n_sites)))
    offsets = []
    for row in range(side):
        for col in range(side):
            if len(offsets) >= n_sites:
                break
            dx = (col - (side - 1) / 2) * spacing
            dy = (row - (side - 1) / 2) * spacing
            offsets.append((float(dx), float(dy)))
    return offsets[:n_sites]


def _get_exposure(exposure, channel):
    """Get exposure for a channel (supports float or dict)."""
    if isinstance(exposure, dict):
        return float(exposure.get(channel, 50.0))
    return float(exposure)
