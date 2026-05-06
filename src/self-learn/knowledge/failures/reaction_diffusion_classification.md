# Failure Analysis: Reaction-Diffusion Pattern Classification

## Challenge Type
Classify reaction-diffusion pattern into types (waves, spirals, spots, stripes, mitosis, coral).

## What Went Wrong
Misclassified mitosis as coral based on a single static frame.
- **Visual**: Dense branching labyrinthine structures — looks like brain coral in a snapshot
- **Ground truth**: "mitosis" — spots that split and divide over time

## Root Cause

Mitosis and coral produce visually similar labyrinthine/branching structures. The static appearance alone is insufficient to distinguish them. The key difference is **temporal behavior**:

- **Mitosis**: Spots actively SPLIT and DIVIDE — new spots appear between frames
- **Coral**: Branching grows slowly from existing tips — no spot splitting

## Correct Classification Approach

### Static features (necessary but not sufficient)
| Pattern  | Coverage | Static Morphology |
|----------|----------|-------------------|
| waves    | 0.1-0.3  | V-shaped/circular traveling fronts |
| spirals  | 0.2-0.4  | Rotating spiral arms |
| spots    | 0.1-0.3  | Round, isolated, static |
| stripes  | 0.3-0.5  | Parallel elongated lines |
| mitosis  | 0.3-0.5  | Spots/blobs that split (looks like coral in snapshot) |
| coral    | 0.4-0.7  | Dense branching from tips |

### Temporal features (critical for disambiguation)
1. Track individual features across 3-4 frames
2. Count connected components at each frame
3. If component count INCREASES significantly → mitosis (spots splitting)
4. If component count is STABLE but structures extend → coral (tip growth)
5. If pattern rotates → spirals
6. If front propagates through field → waves

### Decision tree
```
IF coverage < 0.15: → spots (sparse)
IF temporal_change < 1.0 px mean diff:
    IF elongated features: → stripes
    ELSE: → spots (static)
IF rotating_arms detected: → spirals
IF component_count_increasing: → mitosis
IF coverage > 0.5 AND branching: → coral
IF traveling_front: → waves
```

## Key Lesson
**Always analyze TEMPORAL dynamics for reaction-diffusion classification.** A single snapshot is ambiguous for mitosis vs coral.
