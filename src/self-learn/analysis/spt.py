"""Single-Particle Tracking (SPT) for membrane receptor dynamics.

Implements gap-tolerant track linking, MSD computation, and motion
classification for TIRF-like single-particle microscopy data.

Physical-unit policy (see knowledge/core/strategies/Physical-unit
thresholds.md): ``compute_msd`` and ``analyze_spt`` require a
keyword-only ``pixel_size_um`` from the caller. Recipes pass
``get_config(core).pixel_size_um``; tests pass an explicit calibrated
value. There is no module-level pixel-size default — a forgotten unit
must raise, not silently scale to an arbitrary 100x sensor.

Functions:
    detect_spots          -- Spot detection via blob_log on normalized frame
    link_tracks_gap       -- Gap-tolerant Hungarian track linking
    compute_msd           -- MSD(τ) in physical units from camera px tracks
    classify_track        -- Classify as FREE/CONFINED/DIRECTED via MSD power law
    analyze_spt           -- Full pipeline: detect → link → MSD → classify → report
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.feature import blob_log


# ---- Constants ----------------------------------------------------------

DT_DEFAULT = 0.1              # seconds per frame (100 ms) — algorithmic default

# Classification thresholds (log-log power law α)
ALPHA_CONFINED_MAX = 0.7
ALPHA_DIRECTED_MIN = 1.3


# ---- Spot Detection -----------------------------------------------------

def detect_spots(frame, threshold=0.3, min_sigma=0.8, max_sigma=3.0):
    """Detect fluorescent spots in a single frame via Laplacian of Gaussian.

    Frame is normalized to [0, 1] before detection to make the threshold
    scale-invariant across different illumination conditions.

    Args:
        frame:     2D numpy array (grayscale, any dtype).
        threshold: blob_log threshold on normalized image (0.3 is reliable
                   default — lower gives more spots but more false positives).
                   Use 0.4 for cleaner images, 0.2 for very dim particles.
        min_sigma: Minimum particle radius estimate (camera px).
        max_sigma: Maximum particle radius estimate (camera px).

    Returns:
        (N, 2) array of (row, col) spot centroids in camera pixels.
        Empty array if no spots found.

    Notes:
        - threshold=0.4 may cause heavy blinking (frames with 1-2 spots)
          that breaks track linking. threshold=0.3 is usually more stable.
        - Validate: check min/max spot count ratio over frames (should be < 3x).
    """
    img = np.asarray(frame, dtype=float)
    img_max = img.max()
    if img_max <= 0:
        return np.empty((0, 2))
    img_norm = img / img_max
    blobs = blob_log(img_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                     threshold=threshold)
    if len(blobs) == 0:
        return np.empty((0, 2))
    return blobs[:, :2].copy()  # (row, col)


# ---- Track Linking -------------------------------------------------------

def link_tracks_gap(all_spots, max_dist, max_gap=2, min_len=15):
    """Gap-tolerant track linking using the Hungarian algorithm.

    Handles particle blinking and detection failures by allowing tracks to
    persist through up to max_gap consecutive missing frames.

    Linking rule:
        REJECT_COST = max_dist + 1  (prefer rejection over wrong long match)
        Matches with dist > max_dist are not accepted.

    Args:
        all_spots: List of (N_i, 2) arrays, one per frame. Each row is
                   (row, col) in camera pixels. Empty arrays are OK.
        max_dist:  Maximum allowed displacement per frame (camera pixels).
                   Recommend: 2.5 × σ_step = 2.5 × sqrt(2*D*dt) / px_scale
        max_gap:   Maximum consecutive frames a track can be missing before
                   termination. Default 2; use higher for heavy blinking.
        min_len:   Minimum track length (observed frames only) to keep.

    Returns:
        List of (N_i, 2) numpy arrays, each track's (row, col) positions
        at its observed frames. Gaps are NOT interpolated (raw detections).

    Example:
        >>> spots_per_frame = [detect_spots(f) for f in frames]
        >>> tracks = link_tracks_gap(spots_per_frame, max_dist=25, max_gap=2, min_len=15)
        >>> print(f'{len(tracks)} tracks found')
    """
    REJECT_COST = max_dist + 1
    FAR_COST = REJECT_COST * 3

    # Track storage: {'frames': [fnum, ...], 'positions': [(r,c), ...], 'gap': int}
    tracks = []
    active_ids = []

    for fnum, spots in enumerate(all_spots):
        n_spots = len(spots)

        # No spots this frame: increment all active track gaps
        if n_spots == 0:
            for tid in active_ids:
                tracks[tid]['gap'] += 1
            active_ids = [tid for tid in active_ids
                          if tracks[tid]['gap'] <= max_gap]
            continue

        # No active tracks: start new track for each spot
        if len(active_ids) == 0:
            for spot in spots:
                tracks.append({'frames': [fnum], 'positions': [tuple(spot)], 'gap': 0})
                active_ids.append(len(tracks) - 1)
            continue

        n_active = len(active_ids)
        last_pos = np.array([tracks[tid]['positions'][-1] for tid in active_ids])

        # Pairwise distances: (n_active, n_spots)
        dists = np.sqrt(((last_pos[:, None, :] - spots[None, :, :]) ** 2).sum(axis=2))

        # Build augmented cost matrix for Hungarian
        size = n_active + n_spots
        cost_aug = np.full((size, size), FAR_COST)
        cost_aug[:n_active, :n_spots] = dists
        # Reject columns: active tracks may go unmatched
        for i in range(n_active):
            cost_aug[i, n_spots + i] = REJECT_COST
        # Reject rows: new spots may start new tracks
        for j in range(n_spots):
            cost_aug[n_active + j, j] = REJECT_COST
        # Dummy-dummy block (rows n_active+, cols n_spots+): must be 0 to
        # allow the solver to freely balance excess rows/cols without penalty.
        # If this block is expensive, Hungarian prefers reject paths even when
        # valid close matches exist (leading to 0 tracks!).
        cost_aug[n_active:, n_spots:] = 0

        row_ind, col_ind = linear_sum_assignment(cost_aug)

        matched_active = set()
        matched_spots = set()

        for r, c in zip(row_ind, col_ind):
            if r < n_active and c < n_spots:
                d = dists[r, c]
                if d <= max_dist:
                    tid = active_ids[r]
                    tracks[tid]['frames'].append(fnum)
                    tracks[tid]['positions'].append(tuple(spots[c]))
                    tracks[tid]['gap'] = 0
                    matched_active.add(r)
                    matched_spots.add(c)

        # Unmatched active tracks: increment gap
        for i, tid in enumerate(active_ids):
            if i not in matched_active:
                tracks[tid]['gap'] += 1

        # Terminate tracks that exceeded max_gap
        active_ids = [tid for tid in active_ids if tracks[tid]['gap'] <= max_gap]

        # Start new tracks for unmatched spots
        for j, spot in enumerate(spots):
            if j not in matched_spots:
                tracks.append({'frames': [fnum], 'positions': [tuple(spot)], 'gap': 0})
                active_ids.append(len(tracks) - 1)

    # Return tracks meeting minimum length
    result = []
    for track in tracks:
        n_obs = len(track['frames'])
        if n_obs >= min_len:
            result.append(np.array(track['positions']))
    return result


# ---- MSD Analysis -------------------------------------------------------

def compute_msd(track_xy, *, pixel_size_um, dt=DT_DEFAULT, max_lag_frac=0.5):
    """Compute mean squared displacement vs lag time for one track.

    Delegates core MSD calculation to :func:`~src.core.analysis.diffusion.compute_msd`
    and converts to physical units.

    MSD(τ) = <|r(t+τ) − r(t)|²>  averaged over all valid pairs.

    Args:
        track_xy:      (N, 2) array of (row, col) positions in camera pixels.
        pixel_size_um: Camera pixel size in µm. **Required** keyword-only
            argument — pass ``get_config(core).pixel_size_um`` from a recipe
            or an explicit calibrated value from a test/script. There is no
            silent default: an omitted unit must raise.
        dt:            Frame interval in seconds.
        max_lag_frac:  Max lag as fraction of track length. Default 0.5.

    Returns:
        dict with:
            lags_s:  1D array of lag times in seconds.
            msd_um2: 1D array of MSD values in µm².
    """
    from .diffusion import compute_msd as _msd_core

    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(
            f"pixel_size_um must be > 0, got {pixel_size_um!r}"
        )

    xy = np.asarray(track_xy, dtype=float)
    N = len(xy)
    max_lag = max(1, int(N * max_lag_frac))

    raw = _msd_core(xy, max_lag=max_lag)
    # Convert pixel² → µm², frame lags → seconds
    px_um2 = float(pixel_size_um) ** 2
    return {
        'lags_s': raw['lags'].astype(float) * dt,
        'msd_um2': raw['msd'] * px_um2,
    }


def classify_track(lags_s, msd_um2, n_short_lags=5):
    """Classify track motion type from MSD curve shape.

    Classification via anomalous exponent α (MSD ∝ τ^α):
        FREE:     α ≈ 1.0  (0.7 ≤ α ≤ 1.3) — linear MSD
        CONFINED: α < 0.7  — plateau MSD (corral / lipid raft)
        DIRECTED: α > 1.3  — parabolic MSD (active transport)

    Args:
        lags_s:       1D array of lag times in seconds (from compute_msd).
        msd_um2:      1D array of MSD values in µm² (from compute_msd).
        n_short_lags: Number of short lags for D estimation. Default 5.

    Returns:
        dict with:
            motion:        'FREE', 'CONFINED', or 'DIRECTED'
            alpha:         anomalous diffusion exponent
            D:             diffusion coefficient (µm²/s) from short-lag slope
            r_confinement: corral radius (µm) for CONFINED, else 0.0
    """
    lags = np.asarray(lags_s)
    msd = np.asarray(msd_um2)

    if len(lags) < 5:
        return {'motion': 'FREE', 'alpha': 1.0, 'D': 0.0, 'r_confinement': 0.0}

    # Power-law fit (log-log)
    valid = msd > 0
    if valid.sum() >= 2:
        alpha = float(np.polyfit(np.log(lags[valid]), np.log(msd[valid]), 1)[0])
    else:
        alpha = 1.0

    # Short-lag D from linear MSD slope: MSD = 4*D*t in 2D
    n_fit = min(n_short_lags, len(lags))
    try:
        slope = float(np.polyfit(lags[:n_fit], msd[:n_fit], 1)[0])
        D = max(slope / 4.0, 0.0)
    except Exception:
        D = 0.0

    # Confinement radius from long-lag plateau
    r_confinement = 0.0
    if alpha < ALPHA_CONFINED_MAX:
        n_plateau = max(len(msd) // 3, 1)
        plateau = float(msd[n_plateau:].mean()) if n_plateau < len(msd) else float(msd[-1])
        r_confinement = float(np.sqrt(max(3.0 * plateau / 4.0, 0.0)))

    if alpha < ALPHA_CONFINED_MAX:
        motion = 'CONFINED'
    elif alpha > ALPHA_DIRECTED_MIN:
        motion = 'DIRECTED'
    else:
        motion = 'FREE'

    return {'motion': motion, 'alpha': round(alpha, 3), 'D': round(D, 5),
            'r_confinement': round(r_confinement, 4)}


# ---- Full Pipeline -------------------------------------------------------

def analyze_spt(frames, *, pixel_size_um, dt=DT_DEFAULT,
                blob_threshold=0.3, max_link_px=25, max_gap=2, min_len=15):
    """Full SPT analysis pipeline: detect → link → MSD → classify → aggregate.

    Args:
        frames:          List of 2D grayscale frames (camera pixels).
        pixel_size_um:   Camera pixel size in µm. **Required** keyword-only;
            pass ``get_config(core).pixel_size_um`` from a recipe.
        dt:              Frame interval in seconds.
        blob_threshold:  Spot detection threshold (0.3 recommended).
        max_link_px:     Max displacement per frame for linking (camera px).
                         Rule: 2-3 × σ_step = 2.5 × sqrt(2*D*dt) / pixel_size_um
        max_gap:         Max consecutive missing frames per track.
        min_len:         Minimum track length to keep.

    Returns:
        dict with:
            tracks:            List of (N_i, 2) position arrays (camera px).
            spot_counts:       List of spots per frame.
            track_results:     List of per-track dicts (motion, D, r_conf, alpha).
            D_free:            Median D for FREE tracks (µm²/s).
            D_confined:        Median D for CONFINED tracks (µm²/s).
            r_confinement:     Median confinement radius (µm).
            fraction_free:     Fraction of tracks classified FREE.
            fraction_confined: Fraction of tracks classified CONFINED.
            fraction_directed: Fraction of tracks classified DIRECTED.
            n_tracks:          Total number of valid tracks.
    """
    # Step 1: Detect spots in each frame
    all_spots = []
    for frame in frames:
        all_spots.append(detect_spots(frame, threshold=blob_threshold))
    spot_counts = [len(s) for s in all_spots]

    # Step 2: Link tracks
    tracks = link_tracks_gap(all_spots, max_dist=max_link_px,
                             max_gap=max_gap, min_len=min_len)

    # Step 3: MSD and classification per track
    track_results = []
    D_free_vals, D_conf_vals, r_conf_vals = [], [], []
    for track in tracks:
        msd_result = compute_msd(track, pixel_size_um=pixel_size_um, dt=dt)
        cls = classify_track(msd_result['lags_s'], msd_result['msd_um2'])
        cls['n_frames'] = len(track)
        track_results.append(cls)
        if cls['motion'] == 'FREE' and cls['D'] > 0:
            D_free_vals.append(cls['D'])
        elif cls['motion'] == 'CONFINED':
            if cls['D'] > 0:
                D_conf_vals.append(cls['D'])
            if cls['r_confinement'] > 0:
                r_conf_vals.append(cls['r_confinement'])

    # Step 4: Aggregate
    n_total = len(track_results)
    motions = [t['motion'] for t in track_results]
    n_free = motions.count('FREE')
    n_confined = motions.count('CONFINED')
    n_directed = motions.count('DIRECTED')

    D_free = float(np.median(D_free_vals)) if D_free_vals else 0.0
    D_confined = float(np.median(D_conf_vals)) if D_conf_vals else 0.0
    r_confinement = float(np.median(r_conf_vals)) if r_conf_vals else 0.0
    fraction_free = n_free / max(n_total, 1)
    fraction_confined = n_confined / max(n_total, 1)
    fraction_directed = n_directed / max(n_total, 1)

    return {
        'tracks': tracks,
        'spot_counts': spot_counts,
        'track_results': track_results,
        'D_free': round(D_free, 5),
        'D_confined': round(D_confined, 5),
        'r_confinement': round(r_confinement, 4),
        'fraction_free': round(fraction_free, 3),
        'fraction_confined': round(fraction_confined, 3),
        'fraction_directed': round(fraction_directed, 3),
        'n_tracks': n_total,
        'n_free': n_free,
        'n_confined': n_confined,
        'n_directed': n_directed,
    }


def sigma_step(D_um2_per_s, dt_s, px_um):
    """Expected per-frame step size σ for a diffusing particle.

    Use to validate max_link: max_link should be 2-3 × sigma_step.

    Args:
        D_um2_per_s: Diffusion coefficient (µm²/s).
        dt_s:        Frame interval (seconds).
        px_um:       Camera pixel size (µm).

    Returns:
        σ in camera pixels.

    Example:
        >>> sig = sigma_step(0.1, 0.1, 0.0125)  # D=0.1, 100x sensor, 100ms/frame
        >>> print(f'σ = {sig:.1f} cam px, use max_link ≈ {2.5*sig:.0f}')
    """
    if px_um <= 0:
        raise ValueError(f"px_um must be > 0, got {px_um!r}")
    return float(np.sqrt(2.0 * D_um2_per_s * dt_s) / px_um)
