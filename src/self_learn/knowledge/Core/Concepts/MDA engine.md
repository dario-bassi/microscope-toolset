# PMDAEngine — Custom Hardware Actions

Subclass the MDA engine when you need non-standard hardware operations
that cannot be expressed through MDAEvent properties alone.

## SLM: Use Native `slm_image` (No Subclass Needed)

useq-schema supports SLM masks natively via `MDAEvent.slm_image`:

```python
from useq import MDAEvent, SLMImage
import numpy as np

mask = np.zeros((512, 512), dtype=np.uint8)
mask[200:300, 200:300] = 255

event = MDAEvent(
    channel={"config": "GFP"},
    slm_image=SLMImage(data=mask, device="SLM"),
)
```

Both `run_events()` and `core.mda.run()` handle `slm_image` automatically —
no custom engine subclass needed. See `[[Core/Strategies/SLM optogenetics]]`.

## When to Subclass

- Objective switching mid-sequence (requires recalibration)
- Custom Z-drive or piezo control
- Hardware-triggered acquisition
- Actions that have no MDAEvent property equivalent

## Basic Subclass

```python
from pymmcore_plus.mda import MDAEngine

class MyEngine(MDAEngine):
    def setup_event(self, event):
        """Called BEFORE each acquisition. Configure hardware here."""
        super().setup_event(event)
        # Custom hardware setup

    def exec_event(self, event):
        """Called to actually acquire. Override for custom logic."""
        return super().exec_event(event)

# Register and run
engine = MyEngine(core)
core.mda.set_engine(engine)
core.mda.run(my_sequence)
```

## Objective Switching Example

```python
class MultiScaleEngine(MDAEngine):
    def setup_event(self, event):
        super().setup_event(event)
        target_mag = event.metadata.get('objective', None)
        if target_mag and target_mag != self._current_mag:
            self._mmc.setState('Objective', target_mag)
            self._current_mag = target_mag
            self._mmc.waitForSystem()
```

## Key Points

- `setup_event` runs before acquisition; `exec_event` does the acquisition
- Always call `super()` unless fully replacing the behavior
- `core.mda.set_engine(engine)` — set once, applies to all subsequent runs
- Use `event.metadata` dict to pass custom parameters through MDAEvent
- For SLM: use `slm_image=SLMImage(data=mask, device="SLM")` on the event directly

## Literature

- [[Papers/Pinkard 2021]] — `setup_event` / `exec_event` overrides on `MDAEngine` are the pymmcore-plus equivalent of Pycro-Manager's pre-/post-hardware **acquisition hooks**: the same idea of injecting Python at well-defined points in the acquisition pipeline so non-`MDAEvent` actions (objective swaps, autofocus, custom triggers) compose cleanly into the streaming loop.
