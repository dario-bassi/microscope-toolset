"""Solve harness — standardized challenge startup workflow.

Wraps the mandatory first steps of every challenge:
connect → snap all channels at 10x → characterize → suggest.

Usage:
    from src.core.workflows.solve_harness import challenge_setup
    setup = challenge_setup(core)
    # setup['channels'] = dict of channel→image
    # setup['pixel_size'] = 1.0  (at 10x)
    # setup['sample_info'] = characterize_image result
    # setup['channel_group'] = 'Fake' or 'Channel'

Functions:
    challenge_setup  -- Full startup: snap all channels, characterize, suggest
"""

import numpy as np
from src.core.hardware.core import snap, snap_all_channels, get_pixel_size


def challenge_setup(core, save_dir=None):
    """Run standardized challenge startup workflow.

    Steps:
    1. Detect config group and channels
    2. Snap all channels at 10x
    3. Characterize the sample (intensity stats, structure hints)
    4. Return everything needed for informed analysis

    Args:
        core: CMMCorePlus or proxy instance.
        save_dir: Optional path to save channel images as PNGs.

    Returns:
        dict with:
            channels: dict of channel_name → numpy array.
            channel_names: list of channel names.
            channel_group: config group name ('Fake', 'Channel', etc.).
            pixel_size: float, µm/px at current objective.
            image_shape: tuple (H, W).
            channel_stats: dict of channel_name → {min, max, mean, std}.
            brightest_channel: channel with highest signal range.
    """
    # Detect config group (auto-discover via MicroscopeConfig)
    from src.core.hardware.config import resolve_channel_group
    channel_group = resolve_channel_group(core, None)

    # Get pixel size
    pixel_size = get_pixel_size(core)

    # Snap all channels
    channels = snap_all_channels(core)
    channel_names = list(channels.keys())

    # Channel statistics
    channel_stats = {}
    best_range = 0
    brightest = channel_names[0] if channel_names else None

    for name, img in channels.items():
        img_f = img.astype(np.float64)
        stats = {
            'min': float(img_f.min()),
            'max': float(img_f.max()),
            'mean': float(img_f.mean()),
            'std': float(img_f.std()),
            'range': float(img_f.max() - img_f.min()),
        }
        channel_stats[name] = stats
        if stats['range'] > best_range:
            best_range = stats['range']
            brightest = name

    # Save if requested
    if save_dir is not None:
        from pathlib import Path
        from PIL import Image
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, img in channels.items():
            Image.fromarray(img).save(save_dir / f"{name}.png")

    shape = channels[channel_names[0]].shape if channel_names else (512, 512)

    return {
        'channels': channels,
        'channel_names': channel_names,
        'channel_group': channel_group,
        'pixel_size': pixel_size,
        'image_shape': shape,
        'channel_stats': channel_stats,
        'brightest_channel': brightest,
    }
