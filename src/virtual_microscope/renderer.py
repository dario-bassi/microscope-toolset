"""Optimize rendering using OpenCV"""
import numpy as np
import cv2
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
        self.master_shape = np.array([[-5,-20], [0,-5], [5,-20], [5,20], [0,5], [-5,20]], dtype=np.float32)

    
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
        img = self._apply_filters(img, mode)

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
        """Draw a smooth cell shape using bezier curves or ellipse fitting."""
        n_interp = 100  # More points for smoother appearance
        angles_interp = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
        
        # Interpolate radii to get smooth transitions
        angles_orig = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
        radii_orig = np.linalg.norm(vertices - center, axis=1)
        radii_interp = np.interp(angles_interp, angles_orig, radii_orig, period=2*np.pi)
        
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
            self._draw_smooth_chromatin(cell_img, chromatin_pts_screen, num_strands=46) # 2n = 46, here is double after S phase

        elif cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'S':
            # S phase: DNA replication, uncondensed chromosome
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
            self._draw_smooth_chromatin(cell_img, chromatin_pts_screen, num_strands=92) # 2n = 46, here is double after S phase

        elif cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'G2':
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (150, 60, 60), -1, lineType=cv2.LINE_AA)
            self._draw_smooth_chromatin(cell_img, chromatin_pts_screen, num_strands=92) # 2n = 46, here is double after S phase

        elif cell.cell_mitosis_state == 'Prophase' and cell.cell_cycle_state == 'M':
            # Nucleus dissolve, chromatin condenses into sister chromatine shape
            # Draw condensed chromatin (for simplicity use X shape) without nucleus border
            self._draw_condensed_chromatin(cell_img, cell, num_chromosome=92)

        elif cell.cell_mitosis_state == 'Metaphase'and cell.cell_cycle_state == 'M':
            # Chromatin alignes at cell equator
            self._draw_condensed_chromatin_equator(cell_img, cell, num_chromosome=92)

        elif cell.cell_mitosis_state == 'Anaphase'and cell.cell_cycle_state == 'M':
            # Sister chromatin separate toward cell's poles
            self._draw_condensed_chromatin_polar(cell_img, cell, num_chromosome=92)

        elif cell.cell_mitosis_state == 'Telophase'and cell.cell_cycle_state == 'M':
            # Two nuclei forming at the cell's poles
            # Cell membrane growing, starting to separate
            self._draw_condensed_chromatin_polar(cell_img, cell, num_chromosome=92)
            # Draw two new nuclei
            pole_offset = int(cell_radius * 0.3)
            cv2.circle(cell_img, (nucleus_pos[0] - pole_offset, nucleus_pos[1]),
                       int(nucleus_radius * 0.7), (150, 60, 60), -1, lineType=cv2.LINE_AA)
            cv2.circle(cell_img, (nucleus_pos[0] + pole_offset, nucleus_pos[1]),
                       int(nucleus_radius * 0.7), (150, 60, 60), lineType=cv2.LINE_AA)
            
            # Draw cleavage furrow
            progress = (cell.current_time_life % cell.time_table_mitosis['T']) / (cell.time_table_mitosis['C'] - cell.time_table_mitosis['T'])
            self._draw_cytokenesis_furrow(cell_img, cell, center_screen, vertices, furrow_depth=0.2 + 0.1 * progress)
            
        elif cell.cell_mitosis_state == 'Cytokinesis'and cell.cell_cycle_state == 'M':
            # Draw two separating cells with uncondensed chromatin in nuclei
            progress = (cell.current_time_life % cell.time_table_mitosis['C']) / (cell.time_tot_cycle - cell.time_table_mitosis['C'])

            progress = min(progress, 1.0)

            self._draw_separating_cells(cell_img, cell, center_screen, vertices, progress, camera_offset)
        
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
            #img = cv2.GaussianBlur(img, (3, 3), 1.0)

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
        It render the image according to the current objective selected

        10x -> base image, from the start
        20x -> new image based on 10x image, cropped and rescaled to mantain the dim of the image
        40x -> new image based on 10x image, cropped and rescaled to mantain the dim of the image
        """
        # Compute crop final dim of the image
        if self.objective == 10:
            self.crop_dim = 512 # set default value
            # return original array. No need to change
            return img
        elif self.objective == 20:
            self.crop_dim = 512 * (10 / self.objective)
        else: # 40x case
            self.crop_dim = 512 * (10 / self.objective)

        # Compute coordinates of top left vertice of the image
        x_start = int((self.width - self.crop_dim) / 2)
        y_start = int((self.height - self.crop_dim) / 2)

        
        # slice new image
        crop_img = img[x_start:-x_start, y_start:-y_start]


        # rescale the image to have same dimensions of width and height
        rescaled_img = cv2.resize(crop_img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

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
                        color=(180, 100, 255), thickness=1, # color (100, 80, 150)
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
        cv2.fillPoly(img, [pts], (255, 150, 255), lineType=cv2.LINE_AA)

        # Draw outline
        cv2.polylines(img, [pts], True, (150, 70, 150), 1, lineType=cv2.LINE_AA)
        

    def _draw_condensed_chromatin(self, img: np.ndarray, cell: CellCycleNormal, num_chromosome: int = 46):
        """Draw condensed chromatin, forming a X shape inside the nucleus area."""

        center = np.array(cell.center)
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
    def _draw_condensed_chromatin_equator(self, img: np.ndarray, cell: CellCycleNormal, num_chromosome: int = 46):
        """Draw condensed chromatin at the equator of the cell."""
        center = np.array(cell.center)
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
    def _draw_condensed_chromatin_polar(self, img: np.ndarray, cell: CellCycleNormal, num_chromosome: int = 46):
        """Draw condensed chromatin at the polar pos of the cell."""
        
        center = np.array(cell.center)
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

        cell_radius = cell.base_r
        separation_distance = cell_radius * separation_progress
        
        # Top daughter cell center
        center_top = center + np.array([0, -separation_distance])
        
        # Bottom daughter cell center
        center_bottom = center + np.array([0, separation_distance])
        
        # Scale vertices for separated cells (slightly smaller)
        scale_factor = np.sqrt(0.5)  # Each daughter cell has ~half the area
        vertices_scaled = (vertices - center) * scale_factor
        
        # Draw top cell
        vertices_top = vertices_scaled + center_top
        self._draw_smooth_cell(img, center_top, vertices_top, (100, 100, 200), thickness=-1)
        self._draw_smooth_cell(img, center_top, vertices_top, (0, 0, 0), thickness=2)
        
        # Draw bottom cell
        vertices_bottom = vertices_scaled + center_bottom
        self._draw_smooth_cell(img, center_bottom, vertices_bottom, (100, 100, 200), thickness=-1)
        self._draw_smooth_cell(img, center_bottom, vertices_bottom, (0, 0, 0), thickness=2)

        # draw nuclei with uncondensed chromatin
        nucleus_radius = int(0.4 * cell_radius * scale_factor)

        # Scale chromatin offsets (not absolute pts) for daughter cells
        chromatin_offset_scaled = [offset * scale_factor for offset in cell.chromatin_offset]

        # Top nucleus
        cv2.circle(img, tuple(center_top.astype(int)), nucleus_radius,
                   (150, 60, 60), -1, lineType=cv2.LINE_AA)
        chromatin_top = [(center_top[0] + offset[0] - camera_offset[0], center_top[1] + offset[1] - camera_offset[1]) for offset in chromatin_offset_scaled]
        self._draw_smooth_chromatin(img, chromatin_top, num_strands=46)

        # Bottom nucleus
        cv2.circle(img, tuple(center_bottom.astype(int)), nucleus_radius,
                   (150, 60, 60), -1, lineType=cv2.LINE_AA)
        chromatin_bottom = [(center_bottom[0] + offset[0] - camera_offset[0], center_bottom[1] + offset[1] - camera_offset[1]) for offset in chromatin_offset_scaled]
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