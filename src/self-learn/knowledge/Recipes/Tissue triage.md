# Tissue Triage Protocol Playbook

## When to Use
Multi-well experiments where each well has a different tissue condition. Must survey, classify, then apply condition-specific measurement protocols.

## Workflow

### Phase 1: Survey ALL Wells
1. Set 10x objective
2. Visit every well position
3. At each position, snap ALL channels (BF, nucleus, membrane)
4. Save images for visual inspection
5. **DO NOT classify during survey** — need all data to compare

### Phase 2: Classify Conditions
Compare across wells using both visual and quantitative cues:

| Condition | BF Appearance | Fluorescence | Key Metric |
|-----------|--------------|--------------|------------|
| CONTROL | Normal density, confluent | Normal intensity | Baseline |
| WOUNDED | Horizontal gap/scratch visible | Gap in fluorescence too | Row-wise std drops sharply in gap |
| DRUG-TREATED | Looks like control in BF | DIM nuclei (fraction of control) | Max intensity << other wells |
| OVERGROWTH | Visually much denser | More cells, higher count | Cell count >> control |

**Decision tree:**
1. **Has clear gap/wound?** → WOUNDED
2. **Cell count >> others?** → OVERGROWTH
3. **Nuclear intensity << others?** → DRUG-TREATED
4. **Remaining well** → CONTROL

### Phase 3: Measure Per Condition
- **CONTROL**: Count cells, measure mean nuclear intensity
- **WOUNDED**: Measure wound gap width (px) and wound center
- **DRUG-TREATED**: Mean nuclear intensity, compute suppression ratio = drug_intensity / control_intensity
- **OVERGROWTH**: Count cells, report density (cells per FOV)

## Key Techniques

### Wound Width Measurement
Best approach: **Row-wise standard deviation of BF image**
- Wound region has uniform intensity → low std (< 5)
- Cell region has texture → high std (> 8)
- Find largest contiguous block of rows with std < threshold
- Width = number of rows in that block
- Expected: 60-90 px typically

### Cell Counting
- Threshold nucleus channel (not BF)
- Connected component labeling
- Area filter: 10 < area < 3000 px (remove noise and artifacts)
- Use `skimage.measure.label` + `regionprops`

### Drug Suppression
- Measure mean intensity of detected nuclei (not whole image)
- Ratio = drug_mean / control_mean
- Expected: ~0.30 for 30% suppression
- Use per-nuclei means, not whole-FOV means (avoids background bias)

## Pitfalls
- **Don't classify during survey** — you need all wells to compare relative metrics
- **Drug-treated looks normal in BF** — MUST check fluorescence to distinguish from control
- **Wound detection via threshold** — don't use simple binary; row-wise std is more robust
- **JSON serialization** — convert numpy int64 to Python int before submitting
- **Classification order matters**: identify the easy ones first (wound, overgrowth), then use fluorescence to distinguish drug-treated from control

## Lessons Learned
- Survey ALL wells before classifying — relative comparison is essential
- Drug-treated wells are invisible in BF alone; fluorescence is required
- Row-wise std for wound detection is robust across different wound widths
