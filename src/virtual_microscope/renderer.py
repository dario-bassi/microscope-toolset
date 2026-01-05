"""Optimize rendering using OpenCV"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from .cell_base import CellBase


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
        self.min_opacity = 0.2 # min visbility when far out focus test if visually looks fine

        # render image according to the objective used
        self.crop_dim = 512 # by default 10x crop

    
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
    
    def _get_visible_cells(self, cells: List[CellBase], camera_offset: Tuple[float, float]) -> List[CellBase]:
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
        
        #return smooth_pts
    
    def _draw_cell(self, img: np.ndarray, cell: CellBase, mode: int, 
                   camera_offset: Tuple[float, float], focal_plane: float) -> None:
        """Draw a single cell using OpenCV with proper focal plane effect."""
        # compute blur and opacity using depth of field and z-distance
        kernel_size, opacity = self._compute_blur_and_opacity(cell.z_position, focal_plane)

        # Get vertex postion adjusted for camera
        vertices = (cell.vertices_positions - camera_offset) #* self.zoom
        center_screen = (cell.center - camera_offset) #* self.zoom

        # Adjust cell radius for zoom
        cell_radius = cell.base_r #* self.zoom

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

    