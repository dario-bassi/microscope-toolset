"""Base cell module with core physics and properties."""
import numpy as np
from numba import njit, prange
from numba.types import float64 as no_float64
import numba.types as nbt


# ------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------ #
@njit
def wrap_postion(pos: np.ndarray, width: float, height: float) -> np.ndarray:
    """Wrap position for periodic boundaries conditions."""
    return np.array([pos[0] % width, pos[1] % height])

@njit
def polygon_area(pts: np.ndarray) -> float:
    """Calculate polygon area"""
    x, y = pts[:, 0], pts[:, 1]
    #return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= y[i] * x[j]
    return abs(area) * 0.5

@njit
def calculate_vertices(center: np.ndarray, angles: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Calculate vertex position."""
    n = len(angles)
    vertices = np.empty((n, 2), dtype=np.float64)
    vertices[:, 0] = center[0] + np.cos(angles) * r
    vertices[:, 1] = center[1] + np.sin(angles) * r
    return vertices

@njit
def update_cell_physics(center: np.ndarray, vel: np.ndarray, r: np.ndarray,
                         angles: np.ndarray, base_r: float, area0: float,
                         width: float, height: float, dt: float,
                         friction: float = 3.0, brownian_d: float = 80.0,
                         curvature_relax: float = 0.12, radial_relax: float = 0.08,
                         ruffle_std: float = 0.01, seed: int = 0) -> tuple:
    """Update cell physics"""
    np.random.seed(seed)

    # Brownian motion
    amp = np.random.normal(0, np.sqrt(2 * brownian_d * dt))
    ang = np.random.uniform(0, 2 * np.pi)
    vel += amp * np.array([np.cos(ang), np.sin(ang)])

    # Update position
    center += vel * dt
    center = wrap_postion(center, width, height)

    # Apply friction
    vel *= max(0.0, 1.0 - friction * dt)

    # Membrane ruffling
    r += np.random.normal(0, ruffle_std * base_r, len(r))

    # Curvature relaxation (Laplacian smoothing)
    lap = np.roll(r, -1) + np.roll(r, 1) - 2 * r
    r += curvature_relax * lap + radial_relax * (base_r - r)

    # Constrain radius
    r = np.clip(r, 0.7 * base_r, 1.3 * base_r)

    # conserve area
    vertices = calculate_vertices(center, angles, r)
    area = polygon_area(vertices)
    if area > 0:
        r *= np.sqrt(area0 / area)

    return center, vel, r


@njit
def check_collision(center1: np.ndarray, center2: np.ndarray,
                    r1: np.ndarray, r2: np.ndarray,
                    width: float, height: float) -> tuple:
    """Fast colliision detection and resolution"""
    # Calculate wrapped distance
    dvec = center2 - center1
    dvec[0] -= width * np.round(dvec[0] / width)
    dvec[1] -= height * np.round(dvec[1] / height)

    dist = np.sqrt(dvec[0]**2 + dvec[1]**2)
    if dist == 0:
        return False, center1, center2
    
    overlap = np.max(r1) + np.max(r2) - dist
    if overlap <= 0:
        return False, center1, center2
    
    # resolve collision
    n = dvec / dist
    shift = 0.5 * (overlap + 0.5) * n
    new_center1 = center1 - shift
    new_center2 = center2 + shift

    # Wrap position
    new_center1 = wrap_postion(new_center1, width, height)
    new_center2 = wrap_postion(new_center2, width, height)

    return True, new_center1, new_center2


@njit(parallel=True)
def update_all_cells_parallel(centers: np.ndarray, velocities: np.ndarray,
                              radii: np.ndarray, angles: np.ndarray,
                              base_radii: np.ndarray, areas: np.ndarray,
                              width: float, height: float, dt: float,
                              friction: float = 3.0, brownian_d: float = 80.0) -> None:
    """Update all cells in parallel using Numba prange"""
    n_cells = len(centers)
    for i in prange(n_cells): # parallel loop
        centers[i], velocities[i], radii[i] = update_cell_physics(
            centers[i], velocities[i], radii[i], angles,base_radii[i], areas[i], 
        width, height, dt, friction, brownian_d, seed=i
        )






# ------------------------------------------------------------- #
# Cell
# ------------------------------------------------------------- #
class CellBase:
    """Base cell class with core physics"""
    def __init__(self, width: float, height: float, base_radius: float, vertices: int = 24, seed: int = 0):
        self.width = width
        self.height = height
        self.vertices = vertices
        self.seed = seed

        # Initialize with random variations (# TODO checks later)
        rng = np.random.RandomState(seed)
        self.base_r = base_radius * (0.85 + 0.3 * rng.random())
        self.r = np.full(vertices, self.base_r, dtype=np.float64)
        self.angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
        self.area0 = np.pi * self.base_r ** 2

        # Position and velocity
        self.center = np.array([rng.uniform(0, width), rng.uniform(0, height)], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.z_position: float = 0.0

        # Physics paramters
        self.friction: float = 3.0
        self.brownian_d: float = 80.0
        self.curvature_relax: float = 0.15
        self.radial_relax: float = 0.10
        self.ruffle_std: float = 0.03

        # Fluorescence properties
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence = np.zeros(vertices, dtype=np.float64)

        self._rng = rng


    @property
    def vertices_positions(self) -> np.ndarray:
        """Get absolute vertex positions."""
        return calculate_vertices(self.center, self.angles, self.r)
    
    def update_physics(self, dt: float) -> None:
        """Update cell physics (movement, shape deformation)"""
        self.center, self.vel, self.r = update_cell_physics(
            self.center, self.vel, self.r, self.angles, self.base_r,
            self.area0, self.width, self.height, dt, self.friction, self.brownian_d,
            self.curvature_relax, self.radial_relax, self.ruffle_std, self.seed
            )

    
    def check_collision(self, other: 'CellBase') -> bool:
        """Check and resolve collision with another cell."""
        collided, new_center1, new_center2 = check_collision(
            self.center, other.center, self.r, other.r, self.width, self.height
        )

        if collided:
            self.center = new_center1
            other.center = new_center2
            self.vel[:] = 0
            other.vel[:] = 0

        return collided
    
    def _conserve_area(self) -> None:
        """Conserve cell area after deformation."""
        pts = self.vertices_positions
        area = polygon_area(pts)
        if area > 0:
            self.r *= np.sqrt(self.area0 / area)