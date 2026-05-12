"""Organoid morphometry workflow.

Reusable workflow for 3D organoid analysis: Z-scan navigation,
equatorial plane finding, multi-channel morphometry, and nuclei counting.
Follows the playbook at knowledge/playbooks/organoid.md.

Functions:
    find_equatorial_z  -- Z-scan to find equatorial plane (max cross-section)
    measure_organoid    -- Complete morphometry from multi-channel images
    count_buds          -- Detect and count buds from BF image
    count_wall_nuclei   -- Count nuclei in the wall region only
    organoid_z_profile  -- Cross-section area vs Z for volume estimation
"""

import numpy as np
from scipy import ndimage


def find_equatorial_z(images_and_z, threshold_offset=20):
    """Find the equatorial Z-plane from a BF Z-stack.

    The equatorial plane is where the organoid cross-section area is maximal.
    This corresponds to the widest part of the sphere/ellipsoid.

    Args:
        images_and_z: List of (image, z_position) tuples from a BF Z-scan.
        threshold_offset: Subtracted from p90 background for wall detection.
            Wall = dark ring on bright BF background.

    Returns:
        dict with:
            equatorial_z: Z position of maximum cross-section.
            equatorial_idx: Index into the input list.
            areas: List of filled areas at each Z.
            z_positions: List of Z positions.
    """
    areas = []
    z_positions = []

    for img, z in images_and_z:
        img_f = np.asarray(img, dtype=np.float64)
        bg = np.percentile(img_f, 90)
        wall = img_f < (bg - threshold_offset)
        wall = ndimage.binary_opening(wall, iterations=1)
        filled = ndimage.binary_fill_holes(wall)

        labeled, n = ndimage.label(filled)
        if n == 0:
            areas.append(0)
        else:
            sizes = ndimage.sum(filled, labeled, range(1, n + 1))
            areas.append(int(max(sizes)))
        z_positions.append(float(z))

    best_idx = int(np.argmax(areas))

    return {
        'equatorial_z': z_positions[best_idx],
        'equatorial_idx': best_idx,
        'areas': areas,
        'z_positions': z_positions,
    }


def measure_organoid(bf_image, membrane_image=None, nucleus_image=None,
                     pixel_size=1.0, threshold_offset=20,
                     nuclei_block_size=31, nuclei_min_dist=3):
    """Measure organoid morphometry from equatorial-plane images.

    Combines BF segmentation, membrane ring analysis, and nuclei counting
    into a single comprehensive measurement.

    Args:
        bf_image: 2D brightfield image at equatorial plane.
        membrane_image: Optional membrane/E-cadherin channel image.
        nucleus_image: Optional DAPI/nucleus channel image.
        pixel_size: Microns per pixel.
        threshold_offset: BF wall detection offset from p90 background.
        nuclei_block_size: Block size for adaptive nuclei counting.
        nuclei_min_dist: Min distance for watershed nuclei splitting.

    Returns:
        dict with outer_diameter_um, lumen_diameter_um, wall_thickness_um,
        lumen_present, nuclei_count, eccentricity, outer_area_px.
    """
    from skimage.measure import regionprops

    img_f = np.asarray(bf_image, dtype=np.float64)
    h, w = img_f.shape

    # BF wall segmentation
    bg = np.percentile(img_f, 90)
    wall = img_f < (bg - threshold_offset)
    wall = ndimage.binary_opening(wall, iterations=1)
    filled = ndimage.binary_fill_holes(wall)

    labeled, n = ndimage.label(filled)
    if n == 0:
        return {
            'outer_diameter_um': 0.0,
            'lumen_diameter_um': 0.0,
            'wall_thickness_um': 0.0,
            'lumen_present': False,
            'nuclei_count': 0,
            'eccentricity': 0.0,
            'outer_area_px': 0,
        }

    # Largest connected component = organoid
    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    organoid_mask = labeled == (np.argmax(sizes) + 1)
    organoid_filled = ndimage.binary_fill_holes(organoid_mask)

    props = regionprops(organoid_filled.astype(int))
    if not props:
        return {
            'outer_diameter_um': 0.0,
            'lumen_diameter_um': 0.0,
            'wall_thickness_um': 0.0,
            'lumen_present': False,
            'nuclei_count': 0,
            'eccentricity': 0.0,
            'outer_area_px': 0,
        }

    p = props[0]
    outer_diam = p.equivalent_diameter_area * pixel_size
    outer_area_px = int(p.area)
    eccentricity = float(p.eccentricity)

    # Lumen = hole inside the organoid
    hole = organoid_filled & ~wall
    hole_labeled, n_holes = ndimage.label(hole)
    lumen_present = False
    lumen_diam = 0.0

    if n_holes > 0:
        hole_sizes = ndimage.sum(hole, hole_labeled, range(1, n_holes + 1))
        max_hole = int(max(hole_sizes))
        if max_hole > 50:  # minimum lumen area
            lumen_present = True
            lumen_mask = hole_labeled == (np.argmax(hole_sizes) + 1)
            lumen_props = regionprops(lumen_mask.astype(int))
            if lumen_props:
                lumen_diam = lumen_props[0].equivalent_diameter_area * pixel_size

    # Wall thickness from BF
    wall_thickness_bf = (outer_diam - lumen_diam) / 2 if lumen_present else 0.0

    # Membrane ring analysis (independent measurement)
    wall_thickness_membrane = None
    ring_data = None
    if membrane_image is not None:
        from ..analysis.ring import measure_ring
        ring_data = measure_ring(membrane_image, pixel_size=pixel_size)
        wall_thickness_membrane = ring_data.get('wall_thickness_um', None)

    # Average BF and membrane wall measurements if both available
    if wall_thickness_membrane is not None and wall_thickness_bf > 0:
        wall_thickness = (wall_thickness_bf + wall_thickness_membrane) / 2
    elif wall_thickness_membrane is not None:
        wall_thickness = wall_thickness_membrane
    else:
        wall_thickness = wall_thickness_bf

    # Nuclei counting
    nuclei_count = 0
    if nucleus_image is not None:
        from ..detection.cells import count_nuclei_adaptive
        nuclei_count = count_nuclei_adaptive(
            nucleus_image,
            block_size=nuclei_block_size,
            watershed_min_dist=nuclei_min_dist,
            log_sigma=2.5,
        )

    result = {
        'outer_diameter_um': round(outer_diam, 1),
        'lumen_diameter_um': round(lumen_diam, 1),
        'wall_thickness_um': round(wall_thickness, 1),
        'lumen_present': lumen_present,
        'nuclei_count': nuclei_count,
        'eccentricity': round(eccentricity, 3),
        'outer_area_px': outer_area_px,
    }

    if ring_data is not None:
        result['ring_outer_um'] = ring_data.get('outer_diameter_um', None)
        result['ring_inner_um'] = ring_data.get('inner_diameter_um', None)
        result['ring_wall_um'] = ring_data.get('wall_thickness_um', None)

    return result


def count_buds(bf_image, threshold_offset=20, min_bud_area=50):
    """Count buds on an organoid from a BF image.

    Buds are secondary connected components near the main organoid body.
    Use at 10x or lower mag where the full organoid + buds fit in the FOV.

    Args:
        bf_image: 2D brightfield image at equatorial plane.
        threshold_offset: Subtracted from p90 background for wall detection.
        min_bud_area: Minimum area (pixels) for a connected component to be
            considered a bud (not noise).

    Returns:
        dict with:
            n_buds: Number of buds detected.
            morphology: 'budded' if n_buds > 0, else 'cystic'.
            bud_areas: List of bud areas in pixels.
            bud_centroids: List of (row, col) centroids.
            main_area: Area of the main organoid body.
    """
    from skimage import measure

    img_f = np.asarray(bf_image, dtype=np.float64)
    bg = np.percentile(img_f, 90)
    wall = img_f < (bg - threshold_offset)
    wall = ndimage.binary_opening(wall, iterations=1)
    filled = ndimage.binary_fill_holes(wall)

    labeled, n = ndimage.label(filled)
    if n == 0:
        return {
            'n_buds': 0, 'morphology': 'cystic',
            'bud_areas': [], 'bud_centroids': [], 'main_area': 0,
        }

    props = sorted(measure.regionprops(labeled), key=lambda p: p.area, reverse=True)
    main = props[0]
    main_centroid = main.centroid
    main_diameter = main.equivalent_diameter_area

    buds = []
    for p in props[1:]:
        if p.area < min_bud_area:
            continue
        dist = np.sqrt((p.centroid[0] - main_centroid[0])**2 +
                       (p.centroid[1] - main_centroid[1])**2)
        if dist < main_diameter:  # within 1 diameter of main body
            buds.append(p)

    return {
        'n_buds': len(buds),
        'morphology': 'budded' if buds else 'cystic',
        'bud_areas': [int(b.area) for b in buds],
        'bud_centroids': [(float(b.centroid[0]), float(b.centroid[1])) for b in buds],
        'main_area': int(main.area),
    }


def count_wall_nuclei(nucleus_image, outer_mask, lumen_mask,
                      threshold=12, min_distance=6, min_area=30, max_area=3000):
    """Count nuclei in the organoid wall region only.

    Uses watershed segmentation to split touching nuclei, then filters
    to include only nuclei within the wall (between outer and lumen boundaries).

    Args:
        nucleus_image: 2D DAPI/nucleus channel image.
        outer_mask: Boolean mask of the filled organoid (outer boundary).
        lumen_mask: Boolean mask of the lumen (interior).
        threshold: Intensity threshold for nucleus detection.
        min_distance: Minimum distance between watershed peaks.
        min_area: Minimum nucleus area in pixels.
        max_area: Maximum nucleus area in pixels.

    Returns:
        dict with:
            n_nuclei: Number of nuclei in the wall region.
            n_lumen: Number of nuclei in the lumen (debris).
            centroids: List of (row, col) centroids of wall nuclei.
            areas: List of areas of wall nuclei.
    """
    from skimage import morphology, measure, segmentation
    from skimage.feature import peak_local_max

    nuc = np.asarray(nucleus_image, dtype=np.float64)
    nuc_mask = nuc > threshold
    nuc_clean = morphology.opening(nuc_mask, morphology.disk(1))

    dist = ndimage.distance_transform_edt(nuc_clean)
    coords = peak_local_max(dist, min_distance=min_distance, threshold_abs=3)

    if len(coords) == 0:
        return {'n_nuclei': 0, 'n_lumen': 0, 'centroids': [], 'areas': []}

    markers = np.zeros_like(nuc, dtype=int)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    ws = segmentation.watershed(-dist, markers, mask=nuc_clean)
    ws_props = measure.regionprops(ws)

    wall_band = outer_mask & ~lumen_mask
    h, w = nuc.shape

    wall_nuclei = []
    lumen_nuclei = []
    for p in ws_props:
        if not (min_area < p.area < max_area):
            continue
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        if not (0 <= cy < h and 0 <= cx < w):
            continue
        if wall_band[cy, cx]:
            wall_nuclei.append(p)
        elif lumen_mask[cy, cx]:
            lumen_nuclei.append(p)

    return {
        'n_nuclei': len(wall_nuclei),
        'n_lumen': len(lumen_nuclei),
        'centroids': [(float(p.centroid[0]), float(p.centroid[1])) for p in wall_nuclei],
        'areas': [int(p.area) for p in wall_nuclei],
    }


def organoid_z_profile(images_and_z, pixel_size=1.0, threshold_offset=20):
    """Compute cross-section area vs Z for volume estimation.

    Args:
        images_and_z: List of (image, z_position) tuples from a BF Z-stack.
        pixel_size: Microns per pixel.
        threshold_offset: BF threshold offset.

    Returns:
        dict with:
            z_positions: List of Z values.
            areas_um2: List of cross-section areas in um^2.
            volume_um3: Integrated volume (Riemann sum).
            z_extent_um: Z range where area > 10% of maximum.
            sphericity: V_measured / V_equivalent_sphere.
    """
    equatorial = find_equatorial_z(images_and_z, threshold_offset)
    areas_px = equatorial['areas']
    z_positions = equatorial['z_positions']

    areas_um2 = [a * pixel_size**2 for a in areas_px]

    # Volume by Riemann sum
    z_arr = np.array(z_positions)
    if len(z_arr) > 1:
        z_step = abs(z_arr[1] - z_arr[0])
    else:
        z_step = 1.0
    volume_um3 = sum(a * z_step for a in areas_um2)

    # Z extent: range where area > 10% of max
    max_area = max(areas_um2) if areas_um2 else 0
    if max_area > 0:
        above_10pct = [z for z, a in zip(z_positions, areas_um2) if a > 0.1 * max_area]
        z_extent = max(above_10pct) - min(above_10pct) if above_10pct else 0
    else:
        z_extent = 0

    # Sphericity: compare to equivalent sphere
    if max_area > 0 and z_extent > 0:
        r_eq = np.sqrt(max_area / np.pi)  # radius from max cross-section
        v_sphere = (4 / 3) * np.pi * r_eq**3
        sphericity = min(volume_um3 / v_sphere, 1.0) if v_sphere > 0 else 0
    else:
        sphericity = 0

    return {
        'z_positions': z_positions,
        'areas_um2': areas_um2,
        'volume_um3': round(volume_um3, 1),
        'z_extent_um': round(z_extent, 1),
        'sphericity': round(sphericity, 3),
        'equatorial_z': equatorial['equatorial_z'],
    }
