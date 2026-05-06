"""Runtime microscope configuration discovery.

Discovers hardware capabilities from a pymmcore-plus core instance at runtime,
eliminating hardcoded device names, image sizes, and pixel formulas.

Works with any pymmcore-plus microscope (real or simulated) by querying the
core for available devices, config groups, objectives, and image geometry.

Key classes:
    MicroscopeConfig  -- immutable snapshot of microscope capabilities
    get_config        -- cached config factory (one per core instance)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# MicroscopeConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicroscopeConfig:
    """Immutable snapshot of microscope hardware capabilities.

    Created via ``MicroscopeConfig.from_core(core)`` which queries the core
    for all available devices, config groups, image geometry, and objectives.

    Attributes:
        image_width: Camera image width in pixels.
        image_height: Camera image height in pixels.
        pixel_size_um: Physical pixel size in micrometers.
        n_components: Number of image components (1=gray, 3=RGB, 4=RGBA).
        channel_group: Config group for channels (e.g. "Fake", "Channel"), or None.
        available_channels: List of channel preset names.
        xy_device: XY stage device name, or None if unavailable.
        z_device: Focus device name, or None if unavailable.
        objective_device: Objective/state device name, or None.
        objective_labels: Mapping of state index to label string.
        slm_device: SLM device name, or None.
    """

    image_width: int
    image_height: int
    pixel_size_um: float
    n_components: int = 1

    channel_group: str | None = None
    available_channels: tuple = field(default_factory=tuple)
    xy_device: str | None = None
    z_device: str | None = None
    objective_device: str | None = None
    objective_labels: tuple = field(default_factory=tuple)  # ((idx, label), ...)
    slm_device: str | None = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def image_center_px(self) -> tuple[float, float]:
        """Image center in pixel coordinates (cx, cy)."""
        return self.image_width / 2.0, self.image_height / 2.0

    @property
    def is_rgb(self) -> bool:
        return self.n_components >= 3

    @property
    def fov_width_um(self) -> float:
        """Field of view width in micrometers."""
        return self.image_width * self.pixel_size_um

    @property
    def fov_height_um(self) -> float:
        """Field of view height in micrometers."""
        return self.image_height * self.pixel_size_um

    def magnification_for_label(self, label: str) -> int | None:
        """Parse magnification from an objective label string.

        Handles formats like "10x", "Plan 40x ELWD", "Nikon 100x Oil", etc.
        """
        return _parse_mag_from_label(label)

    def state_for_mag(self, mag: int) -> int | None:
        """Find the objective state index for a given magnification."""
        for idx, label in self.objective_labels:
            parsed = _parse_mag_from_label(label)
            if parsed == mag:
                return idx
        return None

    def current_magnification(self, core) -> int | None:
        """Get current objective magnification by reading core state."""
        if self.objective_device is None:
            return None
        try:
            state = int(core.getState(self.objective_device))
            for idx, label in self.objective_labels:
                if idx == state:
                    mag = _parse_mag_from_label(label)
                    if mag is not None:
                        return mag
        except Exception:
            pass
        # Fallback: try reading the Label property directly
        try:
            label = core.getProperty(self.objective_device, "Label")
            mag = _parse_mag_from_label(str(label))
            if mag is not None:
                return mag
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_core(cls, core, pixel_size_um: float | None = None) -> MicroscopeConfig:
        """Discover microscope configuration from a live core instance.

        Args:
            core: A pymmcore-plus CMMCorePlus (or proxy) instance.
            pixel_size_um: Override pixel size if auto-detection fails.
        """
        # Image geometry
        try:
            width = int(core.getImageWidth())
            height = int(core.getImageHeight())
        except Exception:
            width, height = 512, 512

        try:
            n_comp = int(core.getNumberOfComponents())
        except Exception:
            n_comp = 1

        # Channel group discovery
        channel_group = None
        available_channels: list[str] = []
        skip_groups = {"System", "PixelSize", "Objective", "Real", ""}
        try:
            for grp in core.getAvailableConfigGroups():
                if grp in skip_groups:
                    continue
                try:
                    presets = list(core.getAvailableConfigs(grp))
                except Exception:
                    presets = []
                if presets:
                    channel_group = grp
                    available_channels = presets
                    break
        except Exception:
            pass

        # XY stage
        xy_device = None
        try:
            dev = core.getXYStageDevice()
            if dev:
                xy_device = dev
        except Exception:
            pass

        # Z / focus device
        z_device = None
        try:
            dev = core.getFocusDevice()
            if dev:
                z_device = dev
        except Exception:
            pass

        # Objective device — try common names
        objective_device = None
        objective_labels: list[tuple[int, str]] = []
        for obj_name in ("Objective", "ObjectiveState", "Nosepiece"):
            try:
                n_states = int(core.getNumberOfStates(obj_name))
                if n_states > 0:
                    objective_device = obj_name
                    for i in range(n_states):
                        try:
                            lbl = core.getStateLabel(obj_name, i)
                            objective_labels.append((i, str(lbl)))
                        except Exception:
                            objective_labels.append((i, f"State-{i}"))

                    # If state labels are generic ("State-0"), try Label
                    # property which some proxies expose with real names
                    if all(_parse_mag_from_label(lbl) is None for _, lbl in objective_labels):
                        try:
                            allowed = list(core.getAllowedPropertyValues(obj_name, "Label"))
                            if len(allowed) == len(objective_labels) and any(
                                _parse_mag_from_label(a) for a in allowed
                            ):
                                objective_labels = [(i, allowed[i]) for i in range(len(allowed))]
                        except Exception:
                            pass
                    break
            except Exception:
                continue

        # SLM device
        slm_device = None
        try:
            dev = core.getSLMDevice()
            if dev:
                slm_device = dev
        except Exception:
            pass

        # Pixel size
        if pixel_size_um is not None:
            px_size = float(pixel_size_um)
        else:
            px_size = _discover_pixel_size(
                core,
                objective_labels,
                width,
                objective_device=objective_device,
            )

        return cls(
            image_width=width,
            image_height=height,
            pixel_size_um=px_size,
            n_components=n_comp,
            channel_group=channel_group,
            available_channels=tuple(available_channels),
            xy_device=xy_device,
            z_device=z_device,
            objective_device=objective_device,
            objective_labels=tuple(objective_labels),
            slm_device=slm_device,
        )


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_configs: dict[int, MicroscopeConfig] = {}


def get_config(core) -> MicroscopeConfig:
    """Get or create a MicroscopeConfig for a core instance.

    Caches by ``id(core)`` so repeated calls return the same config.
    Call ``clear_config_cache()`` or ``refresh_config(core)`` if hardware
    changes at runtime (e.g. after switching objectives externally).
    """
    key = id(core)
    if key not in _configs:
        _configs[key] = MicroscopeConfig.from_core(core)
    return _configs[key]


def refresh_config(core) -> MicroscopeConfig:
    """Force re-discovery and cache the new config."""
    key = id(core)
    cfg = MicroscopeConfig.from_core(core)
    _configs[key] = cfg
    return cfg


def clear_config_cache():
    """Clear all cached configs."""
    _configs.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAG_RE = re.compile(r"(\d+)\s*[xX]")


def _parse_mag_from_label(label: str) -> int | None:
    """Extract magnification integer from an objective label.

    Handles: "10x", "Plan 40x ELWD", "Nikon 100x/1.4 Oil", "State-0", etc.
    """
    m = _MAG_RE.search(label)
    if m:
        return int(m.group(1))
    return None


def _discover_pixel_size(core, objective_labels, image_width, objective_device=None) -> float:
    """Try to determine pixel size from core or objective labels.

    Strategy:
    1. Try core.getPixelSizeUm() (works on real microscopes with PixelSize config).
    2. Get CURRENT objective magnification and compute pixel size.
    3. Fall back to 1.0 um/px (safe default for 10x-equivalent).
    """
    # Strategy 1: core reports pixel size
    try:
        px = float(core.getPixelSizeUm())
        if px > 0:
            return px
    except Exception:
        pass

    # Strategy 2: infer from current objective magnification
    mag = _current_mag(core, objective_device, objective_labels)
    if mag is not None and mag > 0:
        return _mag_to_pixel_size(mag, image_width)

    # Strategy 3: default
    return 1.0


def _current_mag(core, objective_device, objective_labels) -> int | None:
    """Get the current objective magnification."""
    if objective_device is None:
        return None
    # Try reading current state index and matching to labels
    try:
        state = int(core.getState(objective_device))
        for idx, label in objective_labels:
            if idx == state:
                mag = _parse_mag_from_label(label)
                if mag is not None:
                    return mag
    except Exception:
        pass
    # Try reading Label property directly
    try:
        label = core.getProperty(objective_device, "Label")
        mag = _parse_mag_from_label(str(label))
        if mag is not None:
            return mag
    except Exception:
        pass
    return None


def _mag_to_pixel_size(mag: int, image_width: int = 512) -> float:
    """Convert magnification to pixel size in micrometers.

    Uses a zoom-factor model: each standard magnification step doubles
    the zoom. At 10x, FOV = image_width micrometers (pixel_size = 1.0).

    Standard mapping (512px sensor):
        10x → 1.0, 20x → 0.5, 40x → 0.25, 100x → 0.125

    For non-standard magnifications, interpolates via the zoom-factor
    model rather than the naive ``10/mag`` formula.
    """
    # Standard zoom factors: each step doubles
    _ZOOM = {10: 1, 20: 2, 40: 4, 60: 6, 100: 8}
    if mag in _ZOOM:
        return image_width / (_ZOOM[mag] * image_width)
    # Fallback for unusual magnifications: 10/mag (approximate)
    return 10.0 / mag
