"""Optimize rendering using OpenCV"""
import numpy as np
import cv2
import random
from typing import List, Tuple, Optional, Sequence
from .cell_base import CellBase
from .cell_cycle import CellCycleNormal


class Renderer:
    """Fast renderer using OpenCV."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width # image dimension - width
        self.height = height # image dimension - height
        self.contrast = 0.55
        self.brightness = 0.78
        self.blur_radius = 3 # blur kernel size
        self.noise_std = 10 # some noise
        self.margins = 100

        self.dof_map = {10: 6.0, 20: 3.0, 40: 1.5} # fine-tune if necessary
        self.objective = 10 # objective value
        self.dof = self.dof_map[self.objective] # fine-tune if necessary

        # Visual Limits
        self.max_blur_radius = 25 # max Gaussian kernel radius, test if visually looks fine
        self.min_opacity = 0.2 # min visibility when far out focus test if visually looks fine

        # render image according to the objective used
        self.crop_dim = 512 # by default 10x crop
        # create master shape for the chromosome
        self.master_shape = np.array([[-1.25,-5], [0,-1.25], [1.25,-5], [1.25,5], [0,1.25], [-1.25,5]], dtype=np.float32)

    
    def render_cells(self, cells: List[CellBase], mode: int = 0,
                     camera_offset: Tuple[float, float] = (0,0),
                     focal_plane: float = 0.0) -> np.ndarray:
        """Render cells to image array using OpenCV."""
        # Create base image (white)
        img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)

        # Get visibile cells
        visible_cells = self._get_visible_cells(cells, camera_offset)

        for cell in visible_cells:
            self._draw_cell(img, cell, mode, camera_offset, focal_plane)

        # apply microscope filters
        img = self._apply_filters(img, mode)

        # crop and rescale base on the objective used
        img = self._crop_and_rescale(img)

        return img
    
    def render_cell_cycle(self, cells: Sequence[CellCycleNormal], mode: int = 0,
                          camera_offset: Tuple[float, float] = (0,0),
                          focal_plane: float = 0.0):
        """Render cell cycle to image array using OpenCV"""
        # Create base image (white)
        img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)

        # Get visibile cells
        visible_cells = self._get_visible_cells(cells, camera_offset)

        for cell in visible_cells:
            self._draw_cell_cycle(img, cell, camera_offset, focal_plane, mode)  # type: ignore

        # apply microscope filters
        #img = self._apply_filters(img, mode)

        # crop and rescale base on the objective used
        img = self._crop_and_rescale(img)

        return img
    
    def _get_visible_cells(self, cells: Sequence[CellBase | CellCycleNormal], camera_offset: Tuple[float, float]) -> Sequence[CellBase | CellCycleNormal]:
        """Filter cells visible in current viewport"""

        margin = self.margins
        vx, vy = camera_offset

        viewport_width = self.width # always leave viewport dimension to 512 
        viewport_height = self.height # always leave viewport dimension to 512

        visible = []
        for cell in cells:
            # only append those cell in or near viewport
            if (vx - margin <= cell.center[0] <= vx + viewport_width + margin and 
                vy - margin <= cell.center[1] <= vy + viewport_height + margin):
                visible.append(cell)
        
        return visible
    
    def _draw_smooth_cell(self, img: np.ndarray, center: np.ndarray, 
                         vertices: np.ndarray,color: tuple, 
                         thickness: int = -1) -> None:
        """Draw a smooth cell shape using cubic spline interpolation for rounder borders."""
        n_interp = 200  # Increased for even smoother appearance
        angles_interp = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
        
        # Get original angles and radii
        angles_orig = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
        radii_orig = np.linalg.norm(vertices - center, axis=1)
        
        # Duplicate first point at the end to handle periodic boundary for spline
        angles_extended = np.append(angles_orig, angles_orig[0] + 2 * np.pi)
        radii_extended = np.append(radii_orig, radii_orig[0])
        
        # Use cubic spline interpolation for much rounder, smoother curves
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(angles_extended, radii_extended, bc_type='periodic')
        radii_interp = cs(angles_interp)
        
        # Ensure radii stay positive
        radii_interp = np.maximum(radii_interp, 0.01)
        
        # Generate smooth vertices
        smooth_verts = np.zeros((n_interp, 2))
        smooth_verts[:, 0] = center[0] + radii_interp * np.cos(angles_interp)
        smooth_verts[:, 1] = center[1] + radii_interp * np.sin(angles_interp)
        smooth_pts = smooth_verts.astype(np.int32)
        
        # Draw the smooth polygon
        if thickness == -1: # filled
            cv2.fillPoly(img, [smooth_pts], color, lineType=cv2.LINE_AA)  # LINE_AA for anti-aliasing
        else:
            cv2.polylines(img, [smooth_pts], True, color, thickness, lineType=cv2.LINE_AA)
        
    
    def _draw_cell_cycle(self, img: np.ndarray, cell: CellCycleNormal,
                         camera_offset: Tuple[float, float], focal_plane: float,
                         mode: int = 0) -> None:
        """Draw a single cell in cell cycle using OpenCV with proper focal plane effect."""
        # compute blur and opacity depth of field and z-distance
        kernel_size, opacity = self._compute_blur_and_opacity(cell.z_position, focal_plane)

        # Get vertex position adjusted for camera (convert to screen space)
        vertices = (cell.vertices_positions - camera_offset)
        center_screen = (cell.center - camera_offset)
        
        # Convert chromatin points to screen space (consistent with vertices)
        chromatin_pts_screen = [(pt[0] - camera_offset[0], pt[1] - camera_offset[1]) for pt in cell.chromatin_pts]

        # Adjust cell radius for zoom
        cell_radius = cell.base_r

        # Skip if center is outside viewport
        if (center_screen[0] < -self.margins or center_screen[0] > self.width + self.margins or
            center_screen[1] < -self.margins or center_screen[1] > self.height + self.margins):
            return
        
        # Handle apoptosis rendering
        if cell.is_dying:
            self._draw_apoptosis_phase(img, cell, center_screen, vertices, chromatin_pts_screen, camera_offset, opacity, kernel_size)
            return
        
        # create temporaly image for the cell
        cell_img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)
        layers = 10 #6 # numbers of layers

        for i in range(layers, 0, -1):
            s = i / layers
            shade = 80 + int(100 * s)
            color = (shade, shade, 255)

            scaled_verts = center_screen + (vertices - center_screen) * s

            self._draw_smooth_cell(cell_img, center_screen, scaled_verts, 
                                    color, thickness=-1)
            
        self._draw_smooth_cell(cell_img, center_screen, vertices,
                            (0, 0, 0), thickness=2)
        
        # Draw nucleus with (dark center)
        nucleus_pos = tuple(center_screen.astype(int))
        nucleus_radius = int(0.4 * cell_radius)

        # Depending on the cell phase draw different type of chromatin or cell size
        # draw uncondensed chromatin
        if cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'G1':
            # Nucleus is intact with uncondesed chromatin
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
            self._draw_smooth_chromatin(cell_img, chromatin_pts_screen, num_strands=20) # 2n = 46, here is double after S phase

        elif cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'S':
            # S phase: DNA replication, uncondensed chromosome
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
            self._draw_smooth_chromatin(cell_img, chromatin_pts_screen, num_strands=40) # 2n = 46, here is double after S phase

        elif cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'G2':
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
            self._draw_smooth_chromatin(cell_img, chromatin_pts_screen, num_strands=40) # 2n = 46, here is double after S phase

        elif cell.cell_mitosis_state == 'Prophase' and cell.cell_cycle_state == 'M':
            # Nucleus dissolve, chromatin condenses into sister chromatine shape
            # Draw condensed chromatin (for simplicity use X shape) without nucleus border
            self._draw_condensed_chromatin(cell_img, cell, camera_offset, num_chromosome=10)

        elif cell.cell_mitosis_state == 'Metaphase'and cell.cell_cycle_state == 'M':
            # Chromatin alignes at cell equator
            self._draw_condensed_chromatin_equator(cell_img, cell, camera_offset, num_chromosome=10)

        elif cell.cell_mitosis_state == 'Anaphase'and cell.cell_cycle_state == 'M':
            # Sister chromatin separate toward cell's poles
            # Cell membrane starts pinching at equator (constriction_progress: 0% → 50%)
            self._draw_condensed_chromatin_polar(cell_img, cell, camera_offset, num_chromosome=10)

        elif cell.cell_mitosis_state == 'Telophase'and cell.cell_cycle_state == 'M':
            # Two nuclei forming at the cell's poles
            # Cell membrane constriction complete (constriction_progress: 50% → 100%)
            # The cell is now deeply pinched with dumbbell/hourglass shape
            self._draw_condensed_chromatin_polar(cell_img, cell, camera_offset, num_chromosome=10)
            # Draw two new nuclei at poles (TOP and BOTTOM along division axis)
            pole_offset = int(cell_radius * 0.5)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] - pole_offset),
                       int(nucleus_radius * 0.7), (150, 60, 60), -1, lineType=cv2.LINE_AA)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] + pole_offset),
                       int(nucleus_radius * 0.7), (150, 60, 60), -1, lineType=cv2.LINE_AA)
            
        elif cell.cell_mitosis_state == 'Cytokinesis'and cell.cell_cycle_state == 'M':
            # Draw pinched cell during cytokinesis
            # Constriction continues beyond 100% as the bridge narrows
            # As constriction_progress increases from 1.0 → 1.3, the bridge decreases in size
            # Once bridge disappears (constriction >= 1.2), division is triggered and
            # sister cell is created in the next state_manager update
            self._draw_condensed_chromatin_polar(cell_img, cell, camera_offset, num_chromosome=10)
        
        # Apply blur based on focal plane
        if kernel_size > 0:
            cell_img = cv2.GaussianBlur(cell_img,
                                        (kernel_size, kernel_size),
                                        kernel_size / 3.0)
            
        cell_opacity = opacity * 1.0
        img[:] = cv2.addWeighted(img, 1.0, cell_img, cell_opacity, 0)
        
    
    def _draw_cell(self, img: np.ndarray, cell: CellBase, mode: int, 
                   camera_offset: Tuple[float, float], focal_plane: float) -> None:
        """Draw a single cell using OpenCV with proper focal plane effect."""
        # compute blur and opacity using depth of field and z-distance
        kernel_size, opacity = self._compute_blur_and_opacity(cell.z_position, focal_plane)

        # Get vertex postion adjusted for camera
        vertices = (cell.vertices_positions - camera_offset)
        center_screen = (cell.center - camera_offset)

        # Adjust cell radius for zoom
        cell_radius = cell.base_r

        # Skip if center is outside viewport
        if (center_screen[0] < -self.margins or center_screen[0] > self.width + self.margins or
            center_screen[1] < -self.margins or center_screen[1] > self.height + self.margins):
            return
        
        # fluorescence mode
        if mode == 0: # brightfield
            # create temporaly image for the cell
            cell_img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)
            layers = 10 #6 # numbers of layers

            for i in range(layers, 0, -1):
                s = i / layers
                shade = 80 + int(100 * s)
                color = (shade, shade, 255)

                scaled_verts = center_screen + (vertices - center_screen) * s

                self._draw_smooth_cell(cell_img, center_screen, scaled_verts, 
                                       color, thickness=-1)
                
            self._draw_smooth_cell(cell_img, center_screen, vertices,
                                (0, 0, 0), thickness=2)
            
            # Draw nucleus with (dark center)
            nucleus_pos = tuple(center_screen.astype(int))
            nucleus_radius = int(0.4 * cell_radius)

            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
            
            # Apply blur based on focal plane
            if kernel_size > 0:
                cell_img = cv2.GaussianBlur(cell_img,
                                            (kernel_size, kernel_size),
                                            kernel_size / 3.0)
                
            cell_opacity = opacity * 1.0
            img[:] = cv2.addWeighted(img, 1.0, cell_img, cell_opacity, 0)

        elif mode == 1: # nucleus fluorescence
            if cell.nucleus_fluorescence > 0:
                print("fluorescence nucleus")
                # Create temporary image for fluorescence
                fluor_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                nucleus_pos = tuple(center_screen.astype(int))
                nucleus_radius = int(0.55 * cell_radius)

                # Drawing glowing nucleus with proper color
                fluorescence_intensity = cell.nucleus_fluorescence * opacity

                # Multiple layers for glow effect
                for i in range(5, 0, -1):
                    radius = int(nucleus_radius * (1.0 + 0.2 * (5 - i)))
                    intensity = int(255 * fluorescence_intensity * (i / 5.0))
                    # red fluorescence mScarlet - BGR format
                    color = (8, 8, intensity)
                    cv2.circle(fluor_img, nucleus_pos, radius, color, 
                                -1, lineType=cv2.LINE_AA)
                
                # Add brighter center spot
                cv2.circle(fluor_img, nucleus_pos, int(nucleus_radius * 0.6),
                           (8, 8, 255), -1, lineType=cv2.LINE_AA)

                base_blur = 21
                if kernel_size > 0:
                    # Add extra blur for out of ocus
                    total_blur = base_blur + (kernel_size * 2)
                    total_blur = total_blur if total_blur % 2 == 1 else total_blur + 1
                    fluor_img = cv2. GaussianBlur(fluor_img, (total_blur, total_blur),
                                                  kernel_size * 0.8)
                else:
                    # Apply blur for glow
                    fluor_img = cv2.GaussianBlur(fluor_img, (base_blur,base_blur), 7)

                # Blend with main image
                img[:] = cv2.add(img, fluor_img)

        elif mode == 2: # membrane fluorescence
            if np.any(cell.membrane_fluorescence > 0):
                print("fluorescence membrane")

                fluor_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                avg_fluorescence = np.mean(cell.membrane_fluorescence) * opacity

                membrane_color = (int(8 * avg_fluorescence), 
                                  int(8 * avg_fluorescence), 
                                  int(255 * avg_fluorescence)) # old 136 8 8

                self._draw_smooth_cell(fluor_img, center_screen, vertices, membrane_color, thickness=-1)

                base_blur = 7
                if kernel_size > 0:
                    # Add extra blur for out-of-focus
                    total_blur = base_blur + (base_blur * 2)
                    total_blur = total_blur if total_blur % 2 == 1 else total_blur + 1
                    fluor_img = cv2.GaussianBlur(fluor_img, (total_blur, total_blur),
                                                 kernel_size * 0.6)
                else:
                    # Apply blur effect for glow
                    fluor_img = cv2.GaussianBlur(fluor_img, (7, 7), 2)

                # Blend with main image
                img[:] = cv2.add(img, fluor_img)

    def _apply_filters(self, img: np.ndarray, mode: int) -> np.ndarray:
        """Apply microscope-like filters using OpenCV"""
        if mode == 0: # brightfield
            # converto to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = img.astype(np.float32)
            img = (img - 128.0) * self.contrast + 128.0
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Final blur realistic appeareance
            kernel_size = 2 * self.blur_radius + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = img.astype(np.float32)
            img = img * self.brightness
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Convert back to BGR for consticency
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # RGB
        else: # Fluorescence mode
            # Slight blur for realistic fluorescence (to check!)
            kernel_size = 2 * self.blur_radius + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1.0)

        
        return img
    
    def set_objective(self, mag: int, dof: float) -> None:
        """Set the current objective and its depth of field"""
        self.objective = mag # set magnitude objective
        self.dof = dof # set depth of field


    def _compute_blur_and_opacity(self, cell_z: float, focal_plane: float) -> Tuple[int, float]:
        """
        Convert a cell z-position and focal_plane (both in microns) to
        a blur kernel radius (odd integer kernel = 2*r+1) and opacity scalar.

        Returns (kernel_size, opacity) where kernel_size is 0 for no blur.

        """
        # Calculate focus effects based on z-distance from focal plane
        # z positions from focal plane
        z_dist_um = abs(cell_z - focal_plane)

        # Inside DOF -> sharp
        if z_dist_um <= self.dof:
            return 0, 1.0

        # Out of focus amount
        out_um = z_dist_um - self.dof

        # blur amoun increases with distance from focal plane
        # Map out_um to blur radius (pixel radius)
        # Tunable mapping: gentle growth near DOF, faster for larger distances
        # normalized by DOF to be objective-aware
        norm = out_um / max(1.0, self.dof * 4.0)
        radius = int(min(self.max_blur_radius, (norm ** 0.75) * self.max_blur_radius))

        # Ensure odd kernel = 2*radius+1 later; return radius as kernel radius
        if radius > 0:
            kernel = radius if radius % 2 == 1 else radius + 1
        else:
            kernel = 0

        # opacity decreases with distance
        opacity = max(self.min_opacity, 1.0 / (1.0 + 0.12 * (out_um / max(1.0, self.dof))))

        return kernel, float(opacity)
    
    def _crop_and_rescale(self, img: np.ndarray) -> np.ndarray:
        """
        Render the image according to the current objective selected.

        Higher magnification objectives crop a smaller region from the center
        of the base render and rescale to the full viewport size.

        The stage position represents the center of the FOV (parcentric):
            world = stage + (pixel - 256) * pixel_size_um
        where pixel_size_um = 10 / objective_mag.
        """
        # Compute crop size: higher mag → smaller crop
        if self.objective == 10:
            self.crop_dim = 512
            return img

        self.crop_dim = int(512 * (10 / self.objective))

        # Crop from center so switching objectives keeps the same FOV center
        mid = 256
        half = self.crop_dim // 2
        crop_img = img[mid - half:mid + half, mid - half:mid + half]

        # Rescale back to full viewport dimensions
        rescaled_img = cv2.resize(crop_img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return rescaled_img
    

    def _draw_smooth_chromatin(self, img: np.ndarray, control_points: list[tuple[float, float]], num_strands: int = 46):
        """Draw uncondensed chromatin as smooth strands using Bezier curves."""

        if not control_points or len(control_points) < 2:
            return
        # Each chromatin strand has a curve shape through control points
        num_points_per_strand = 30

        for strand_idx in range(num_strands):

            # Vary strand apperance slightly
            offset_angle = (strand_idx / num_strands) * 2 * np.pi
            curve_pts = []

            for t in np.linspace(0, 1, num_points_per_strand):
                # Generate smooth points between them (Quadratic Bezier)
                p0 = np.array(control_points[0])
                p1 = np.array(control_points[1 % len(control_points)])
                p2 = np.array(control_points[2 % len(control_points)])
                # Formula: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
                res = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                # append the points
                curve_pts.append(res.astype(np.int32))

            # 3. Draw the strand
            curve_pts = np.array(curve_pts).reshape((-1, 1, 2))
            cv2.polylines(img, [curve_pts], isClosed=False, 
                        color=(63, 0, 0), thickness=1,  # Thin chromatin strands
                        lineType=cv2.LINE_AA) # LINE_AA is crucial for smoothness
            

    
    def _get_transformed_point(self, points: np.ndarray, center: np.ndarray, angle: float, scale: float) -> np.ndarray:
        """Transform a point: scale and rotate around center, then translate."""

        # Scale all points
        scaled = points * scale

        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Apply rotation to all points at once
        rotated = scaled @ rot_matrix.T

        return rotated + center
    
    def _draw_chromosome_x(self, img: np.ndarray, center: np.ndarray,
                           rotation: float, scale: float):
        """Draw a chromosome at given position and rotation."""
        # Transform all points of master shape at once
        transformed_points = self._get_transformed_point(self.master_shape, center, rotation, scale)

        # Convert to int32 for OpenCV
        pts = transformed_points.astype(np.int32)

        # Draw as filled polygon
        cv2.fillPoly(img, [pts], (200, 0, 200), lineType=cv2.LINE_AA)  # Purple - visible over red nucleus

        # Draw outline
        cv2.polylines(img, [pts], True, (150, 0, 150), 1, lineType=cv2.LINE_AA)  # Darker purple outline
        

    def _draw_condensed_chromatin(self, img: np.ndarray, cell: CellCycleNormal, camera_offset: Tuple[float, float], num_chromosome: int = 46):
        """Draw condensed chromatin, forming a X shape inside the nucleus area."""

        center = np.array(cell.center) - np.array(camera_offset)
        nucleus_radius = int(0.4 * cell.base_r)

        # Draw chromosomes scattered throught the nucleus
        for i in range(num_chromosome):
            # Random position within the nucleus
            angle = (i / num_chromosome) * 2 * np.pi + np.random.uniform(-0.3, 0.3)
            radius = np.random.uniform(0, nucleus_radius * 0.8)

            chrom_x = center[0] + radius * np.cos(angle)
            chrom_y = center[1] + radius * np.sin(angle)
            chrom_center = np.array([chrom_x, chrom_y])

            # Rotation angle for the X shape
            rotation = np.random.uniform(0, 2 * np.pi)

            self._draw_chromosome_x(img, chrom_center, rotation, scale=1.0)


    # move condensed chromatin to the center along an axis
    def _draw_condensed_chromatin_equator(self, img: np.ndarray, cell: CellCycleNormal, camera_offset: Tuple[float, float], num_chromosome: int = 46):
        """Draw condensed chromatin at the equator of the cell."""
        center = np.array(cell.center) - np.array(camera_offset)
        cell_radius = cell.base_r

        # Metaphase plate thickness
        plate_thickness = int(cell_radius * 0.2)

        # Arrange chromosome in two row along the equator
        chromosome_per_row = num_chromosome // 2
        spacing = (cell_radius * 1.6) / (chromosome_per_row + 1)

        # Top row (y = center - offset)
        for i in range(chromosome_per_row):
            x_pos = center[0] - cell_radius * 0.8 + (i + 1) *spacing
            y_pos = center[1] - plate_thickness // 2
            chrom_center = np.array([x_pos, y_pos])

            # Chromosome aligend vertically
            rotation = np.pi / 2
            self._draw_chromosome_x(img, chrom_center, rotation, scale=1.0)

        # Bottom row (y = center + offset)
        for i in range(num_chromosome - chromosome_per_row):
            x_pos = center[0] - cell_radius * 0.8 + (i + 1) *spacing
            y_pos = center[1] - plate_thickness // 2
            chrom_center = np.array([x_pos, y_pos])

            # Chromosome aligend vertically
            rotation = np.pi / 2
            self._draw_chromosome_x(img, chrom_center, rotation, scale=1.0)


    # separate half of the condensed chromatin towards the polar pos in the cell
    # and increase the cell size to double (two alonged cell)
    def _draw_condensed_chromatin_polar(self, img: np.ndarray, cell: CellCycleNormal, camera_offset: Tuple[float, float], num_chromosome: int = 46):
        """Draw condensed chromatin at the polar pos of the cell."""
        
        center = np.array(cell.center) - np.array(camera_offset)
        cell_radius = cell.base_r

        # Distance of poles from center
        pole_distance = cell_radius * 0.5

        # Separate into two groups moving towards opposite poles
        chromosomes_per_pole = num_chromosome // 2

        # North pole (top)
        pole_north = np.array([center[0], center[1] - pole_distance])
        for i in range(chromosomes_per_pole):
            # Scatter around pole
            angle = (i / max(1, chromosomes_per_pole)) * 2 * np.pi
            radius = np.random.uniform(0, cell_radius * 0.3)
            
            chrom_x = pole_north[0] + radius * np.cos(angle)
            chrom_y = pole_north[1] + radius * np.sin(angle)
            chrom_center = np.array([chrom_x, chrom_y])
            
            rotation = np.random.uniform(0, 2 * np.pi)
            self._draw_chromosome_x(img, chrom_center, rotation, scale=0.9)
        
        # South pole (bottom)
        pole_south = np.array([center[0], center[1] + pole_distance])
        for i in range(num_chromosome - chromosomes_per_pole):
            # Scatter around pole
            angle = (i / max(1, num_chromosome - chromosomes_per_pole)) * 2 * np.pi
            radius = np.random.uniform(0, cell_radius * 0.3)
            
            chrom_x = pole_south[0] + radius * np.cos(angle)
            chrom_y = pole_south[1] + radius * np.sin(angle)
            chrom_center = np.array([chrom_x, chrom_y])
            
            rotation = np.random.uniform(0, 2 * np.pi)
            self._draw_chromosome_x(img, chrom_center, rotation, scale=0.9)


    def _draw_cytokenesis_furrow(self, img: np.ndarray, cell: CellCycleNormal,
                                 center: np.ndarray, vertices: np.ndarray,
                                 furrow_depth: float = 0.3) -> None:
        """Draw the cleavage furrow during telophase/cytokenesis"""
        # Calculate perpendicular direction to division axis
        # For simplicity, assuming vertical division

        # Draw inward pinching at equator
        equator_y = center[1]
        left_bound = int(center[0] - np.max(np.abs(vertices[:, 0] - center[0])))
        right_bound = int(center[0] + np.max(np.abs(vertices[:, 0] - center[0])))

        # Deepening furrow as cytokenesis progress
        furrow_depth_px = int(cell.base_r * furrow_depth)

        # draw dark line to show cleavage
        cv2.line(img, (left_bound, int(equator_y)), (right_bound, int(equator_y)),
                 (50, 50, 50), 2, lineType=cv2.LINE_AA)
        
        # Draw subtle shading above and below furrow
        pts_top = np.array([
            [left_bound, int(equator_y - furrow_depth_px)],
            [right_bound, int(equator_y - furrow_depth_px)],
            [right_bound, int(equator_y)],
            [left_bound, int(equator_y)]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts_top], (40, 40, 80), lineType=cv2.LINE_AA)

    
    def _draw_separating_cells(self, img: np.ndarray, cell: CellCycleNormal,
                               center: np.ndarray, vertices: np.ndarray,
                               separation_progress: float = 0.5, camera_offset: Tuple[float, float] = (0, 0)) -> None:
        """Draw two separating cells during cytokenesis with connection bridge."""
        # separation_progress: 0.0 = fully connected, 1.0 = fully separated
        
        # Use original radius (unconstricted) to calculate proper daughter cell size
        # This avoids issues from cytokinesis constriction deforming the cells
        cell_radius = cell.original_base_r if hasattr(cell, 'original_base_r') else cell.base_r
        separation_distance = cell_radius * separation_progress
        
        # Top daughter cell center
        center_top = center + np.array([0, -separation_distance])
        
        # Bottom daughter cell center
        center_bottom = center + np.array([0, separation_distance])
        
        # Reconstruct vertices from original radius (avoid using constricted vertices)
        # This ensures daughter cells have proper circular shape, not pinched/deformed
        scale_factor = np.sqrt(0.5)  # Each daughter cell has ~half the area
        if hasattr(cell, 'original_r'):
            # Use original radii to get proper unconstricted shape
            angles = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
            reconstructed_radii = cell.original_r * scale_factor
            vertices_scaled = np.zeros_like(vertices)
            for i, angle in enumerate(angles):
                vertices_scaled[i] = np.array([
                    reconstructed_radii[i] * np.cos(angle),
                    reconstructed_radii[i] * np.sin(angle)
                ])
        else:
            # Fallback: use current vertices but scaled
            vertices_scaled = (vertices - center) * scale_factor
        
        # Draw top cell with blue gradient layers
        # Reduce brightness for daughter cells (two cells rendered = reduce to ~half brightness each)
        vertices_top = vertices_scaled + center_top
        layers = 10
        brightness_reduction = 0.6  # Reduce to 60% brightness to compensate for two cells
        for i in range(layers, 0, -1):
            s = i / layers
            shade = int((80 + int(100 * s)) * brightness_reduction)
            color = (shade, shade, int(255 * brightness_reduction))
            scaled_verts_top = (vertices_scaled - center_top) * (i / layers) + center_top
            self._draw_smooth_cell(img, center_top, scaled_verts_top, color, thickness=-1)
        self._draw_smooth_cell(img, center_top, vertices_top, (0, 0, 0), thickness=2)
        
        # draw nuclei with uncondensed chromatin
        nucleus_radius = int(0.4 * cell_radius * scale_factor)

        # Scale chromatin offsets (not absolute pts) for daughter cells
        chromatin_offset_scaled = [(offset[0] * scale_factor, offset[1] * scale_factor) for offset in cell.chromatin_offset]

        # Top nucleus (always drawn - it's in viewport)
        cv2.circle(img, tuple(center_top.astype(int)), nucleus_radius,
                   (150, 60, 60), -1, lineType=cv2.LINE_AA)
        # center_top is already in screen space, so don't subtract camera_offset again
        chromatin_top = [(center_top[0] + offset[0], center_top[1] + offset[1]) for offset in chromatin_offset_scaled]
        self._draw_smooth_chromatin(img, chromatin_top, num_strands=46)

        # Only draw bottom cell if it's within visible viewport bounds
        # Skip if center_bottom is outside viewport to avoid shadow artifacts
        if not (center_bottom[1] < -self.margins or center_bottom[1] > self.height + self.margins):
            vertices_bottom = vertices_scaled + center_bottom
            for i in range(layers, 0, -1):
                s = i / layers
                shade = int((80 + int(100 * s)) * brightness_reduction)
                color = (shade, shade, int(255 * brightness_reduction))
                scaled_verts_bottom = (vertices_scaled - center_bottom) * (i / layers) + center_bottom
                self._draw_smooth_cell(img, center_bottom, scaled_verts_bottom, color, thickness=-1)
            self._draw_smooth_cell(img, center_bottom, vertices_bottom, (0, 0, 0), thickness=2)

            # Bottom nucleus
            cv2.circle(img, tuple(center_bottom.astype(int)), nucleus_radius,
                       (150, 60, 60), -1, lineType=cv2.LINE_AA)
            # center_bottom is already in screen space, so don't subtract camera_offset again
            chromatin_bottom = [(center_bottom[0] + offset[0], center_bottom[1] + offset[1]) for offset in chromatin_offset_scaled]
            self._draw_smooth_chromatin(img, chromatin_bottom, num_strands=46)
            
            # Draw connecting bridge (narrowing as separation progresses)
            if separation_progress < 0.95:
                bridge_width = int(cell_radius * 0.3 * (1.0 - separation_progress))
                bridge_color = (120, 120, 180)
                
                top_connect = center_top + np.array([0, cell_radius * scale_factor])
                bottom_connect = center_bottom + np.array([0, -cell_radius * scale_factor])
                
                cv2.line(img, tuple(top_connect.astype(int)), 
                        tuple(bottom_connect.astype(int)),
                        bridge_color, max(1, bridge_width), lineType=cv2.LINE_AA)
            

    def _draw_apoptosis_phase(self, img: np.ndarray, cell: CellCycleNormal, 
                              center_screen: np.ndarray, vertices: np.ndarray,
                              chromatin_pts_screen: list, camera_offset: Tuple[float, float],
                              opacity: float, kernel_size: int) -> None:
        """Main dispatcher for apoptotic cell rendering based on phase."""
        cell_img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)
        
        if cell.apoptosis_death_phase == 'Shrinkage':
            self._draw_shrinkage_apoptosis(cell_img, cell, center_screen, vertices, chromatin_pts_screen, camera_offset)
        elif cell.apoptosis_death_phase == 'Blebbing':
            self._draw_blebbing_apoptosis(cell_img, cell, center_screen, vertices, chromatin_pts_screen)
        elif cell.apoptosis_death_phase == 'Apoptotic bodies':
            self._draw_apoptotic_bodies_phase(cell_img, cell, center_screen, chromatin_pts_screen, camera_offset)
        elif cell.apoptosis_death_phase == 'Phagocytosis':
            self._draw_phagocytosis_apoptosis(cell_img, cell, center_screen)
        
        # Apply blur and opacity
        if kernel_size > 0:
            cell_img = cv2.GaussianBlur(cell_img, (kernel_size, kernel_size), kernel_size / 3.0)
        
        apoptosis_opacity = self._get_apoptosis_opacity(cell) * opacity
        img[:] = cv2.addWeighted(img, 1.0, cell_img, apoptosis_opacity, 0)

    def _draw_shrinkage_apoptosis(self, img: np.ndarray, cell: CellCycleNormal,
                                  center_screen: np.ndarray, vertices: np.ndarray,
                                  chromatin_pts_screen: list, camera_offset: Tuple[float, float]) -> None:
        """Draw shrinking cell with condensed chromatin visible inside."""
        # Get shrinkage factor and apply to vertices
        shrinkage_factor = cell._get_current_shrinkage_factor()
        vertices_shrunk = center_screen + (vertices - center_screen) * shrinkage_factor
        
        # Draw shrinking cell membrane with blue gradient layers (same as normal cell cycle)
        layers = 10
        for i in range(layers, 0, -1):
            s = i / layers * shrinkage_factor
            shade = 80 + int(100 * s / shrinkage_factor) if shrinkage_factor > 0 else 80
            color = (shade, shade, 255)
            scaled_verts = center_screen + (vertices_shrunk - center_screen) * (i / layers)
            self._draw_smooth_cell(img, center_screen, scaled_verts, color, thickness=-1)
        
        self._draw_smooth_cell(img, center_screen, vertices_shrunk, (0, 0, 0), thickness=2)
        
        # Draw shrinking nucleus
        nucleus_radius = int(0.4 * cell.base_r * shrinkage_factor)
        nucleus_pos = tuple(center_screen.astype(int))
        cv2.circle(img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
        
        # Draw condensed chromatin (nuclear material becoming more compact)
        self._draw_condensed_chromatin(img, cell, camera_offset, num_chromosome=10)

    def _draw_blebbing_apoptosis(self, img: np.ndarray, cell: CellCycleNormal,
                                 center_screen: np.ndarray, vertices: np.ndarray,
                                 chromatin_pts_screen: list) -> None:
        """Draw cell with membrane blebs (8-10 protrusions) forming."""
        blebbing_progress = (cell.death_timer - cell.time_table_apoptois['Shrinkage']) / \
                            (cell.time_table_apoptois['Blebbing'] - cell.time_table_apoptois['Shrinkage'])
        
        # Draw base cell at the final shrinkage size (0.85 = 15% shrinkage)
        # Use the actual shrunk base_r from the cell physics, not hardcoded factor
        shrinkage_factor = cell.base_r / (cell.original_base_r_for_apoptosis if hasattr(cell, 'original_base_r_for_apoptosis') else cell.base_r)
        vertices_blebbed = center_screen + (vertices - center_screen) * shrinkage_factor
        
        # Draw cell with blue gradient layers (same as normal cell cycle)
        layers = 10
        for i in range(layers, 0, -1):
            s = i / layers * shrinkage_factor
            shade = 80 + int(100 * s / shrinkage_factor) if shrinkage_factor > 0 else 80
            color = (shade, shade, 255)
            scaled_verts = center_screen + (vertices_blebbed - center_screen) * (i / layers)
            self._draw_smooth_cell(img, center_screen, scaled_verts, color, thickness=-1)
        
        # Generate and draw blebs (8-10) in blue color from gradient
        bleb_positions = self._generate_bleb_points(center_screen, vertices_blebbed, 
                                                    num_blebs=int(8 + 2 * blebbing_progress))
        
        bleb_color = (130, 130, 255)  # Blue from gradient range
        for bleb_pos, bleb_radius in bleb_positions:
            cv2.circle(img, tuple(bleb_pos.astype(int)), bleb_radius, bleb_color, -1, lineType=cv2.LINE_AA)
        
        # Draw membrane outline
        self._draw_smooth_cell(img, center_screen, vertices_blebbed, (0, 0, 0), thickness=1)
        
        # Draw nucleus inside (intact but shrinking)
        nucleus_radius = int(0.35 * cell.base_r * 0.7)
        nucleus_pos = tuple(center_screen.astype(int))
        cv2.circle(img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)

    def _draw_apoptotic_bodies_phase(self, img: np.ndarray, cell: CellCycleNormal,
                                     center_screen: np.ndarray, chromatin_pts_screen: list,
                                     camera_offset: Tuple[float, float]) -> None:
        """Draw scattered apoptotic bodies with fragmented nucleus inside."""
        if not hasattr(cell, 'apoptotic_body_positions'):
            return
        
        body_radius = int(cell.base_r * 0.25)  # Each body is small
        body_color = (130, 130, 255)  # Blue from gradient range
        
        # Draw apoptotic bodies in blue
        for body_pos in cell.apoptotic_body_positions:
            body_screen = np.array([body_pos[0] - camera_offset[0], body_pos[1] - camera_offset[1]])
            cv2.circle(img, tuple(body_screen.astype(int)), body_radius, body_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, tuple(body_screen.astype(int)), body_radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def _draw_phagocytosis_apoptosis(self, img: np.ndarray, cell: CellCycleNormal,
                                     center_screen: np.ndarray) -> None:
        """Draw fading apoptotic bodies with decreasing opacity."""
        if not hasattr(cell, 'apoptotic_body_positions'):
            return
        
        phagocytosis_progress = (cell.death_timer - cell.time_table_apoptois['Apoptotic bodies']) / \
                               (cell.max_death_timer - cell.time_table_apoptois['Apoptotic bodies'])
        
        body_radius = int(cell.base_r * 0.25)
        # Fade from blue gradient color (130, 130, 255) to black
        fade_factor = 1.0 - phagocytosis_progress
        fade_b = int(130 * fade_factor)
        fade_g = int(130 * fade_factor)
        fade_r = int(255 * fade_factor)
        
        for body_pos in cell.apoptotic_body_positions:
            cv2.circle(img, tuple(np.array(body_pos).astype(int)), body_radius, 
                      (fade_b, fade_g, fade_r), -1, lineType=cv2.LINE_AA)

    def _generate_bleb_points(self, center_screen: np.ndarray,
                             vertices: np.ndarray, num_blebs: int = 8) -> list:
        """Generate bleb protrusion positions on membrane.
        
        Returns list of (position, radius) tuples for each bleb.
        """
        blebs = []
        
        # Get vertices as angles and distances
        angles = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
        radii_orig = np.linalg.norm(vertices - center_screen, axis=1)
        max_radius = np.max(radii_orig)
        
        for i in range(num_blebs):
            # Random position on membrane
            angle = random.uniform(0, 2 * np.pi)
            
            # Bleb extends outward from membrane
            base_radius = max_radius * 0.9
            bleb_extension = max_radius * 0.4  # Blebs can extend 40% beyond radius
            bleb_radius = int(max_radius * 0.15)  # Size of each bleb
            
            # Position on membrane + extension
            bleb_distance = base_radius + bleb_extension * random.uniform(0.3, 1.0)
            
            bleb_x = center_screen[0] + bleb_distance * np.cos(angle)
            bleb_y = center_screen[1] + bleb_distance * np.sin(angle)
            bleb_pos = np.array([bleb_x, bleb_y])
            
            blebs.append((bleb_pos, bleb_radius))
        
        return blebs

    def _get_apoptosis_opacity(self, cell: CellCycleNormal) -> float:
        """Calculate opacity for current apoptosis phase."""
        if cell.apoptosis_death_phase == 'Phagocytosis':
            # Fade out during phagocytosis
            phagocytosis_progress = (cell.death_timer - cell.time_table_apoptois['Apoptotic bodies']) / \
                                   (cell.max_death_timer - cell.time_table_apoptois['Apoptotic bodies'])
            return 1.0 - phagocytosis_progress
        else:
            # Full opacity for earlier phases
            return 1.0
