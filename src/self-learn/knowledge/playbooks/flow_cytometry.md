# Flow Cytometry Analysis

## When to Use
Flow cytometry measures physical and chemical properties of cells as they flow
past laser interrogation points. Not a microscopy imaging modality — analysis
is per-event (per-cell), not spatial.

## Key Concepts
- Each event = one cell passing the detector
- **Forward scatter (FSC)**: proportional to cell SIZE
- **Side scatter (SSC)**: proportional to cell GRANULARITY/complexity
- **Fluorescence channels**: marker-specific (e.g., FITC, PE, APC)
- Events from different channels are NOT co-registered (each measures a different cell)

## Population Gating (typical peripheral blood)
- **Debris**: very small FSC, variable SSC → exclude first
- **Lymphocytes**: small FSC, low SSC → 25-55% of WBCs
- **Granulocytes (neutrophils)**: medium FSC, high SSC → 25-50%
- **Monocytes**: large FSC, medium SSC → 5-17%

## Analysis Workflow
1. **Scatter gating**: Plot FSC vs SSC to identify major populations
2. **Debris exclusion**: Remove very small events (noise)
3. **Fluorescence thresholding**: For each marker, determine positive/negative cutoff
4. **Population fractions**: Count events in each gate
5. **Cross-reference**: e.g., CD3+ fraction among lymphocytes = CD3_positive / lymphocyte_count

## Detection from Image Representation
When flow cytometry data is presented as images (e.g., dot plots, histograms):
- Background subtraction with row-wise median for gradient correction
- Connected components for cell detection in scatter images
- Bright spots above threshold for fluorescence-positive cells
- Cell size from object radius (FSC proxy)
- Cell granularity from center darkness (SSC proxy)

## Common Pitfalls
- **Cannot co-register across channels**: Each acquisition captures different cells
- **Ring artifacts in scatter**: Cells may appear as rings (bright edge + dark center)
  — use morphological closing before connected components
- **Background gradient**: Flow band may have intensity gradient; use local background subtraction
- **Gating bias**: If marker-positive fraction > gated population fraction,
  the gate is too restrictive
