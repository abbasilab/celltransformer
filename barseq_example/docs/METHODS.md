# Methods and scope of the BARseq example

## Public data and cohort

The input is Chen, Fischer and colleagues' BARseq **pilot hemisphere**, Mendeley
Data `8bhhk7c5n9`, version 1. `DOWNLOAD_MANIFEST.json` pins all four URLs, expected
sizes and SHA256 values. Required downloads total **826,716,670 bytes**. These are
processed measurements and author registrations, not microscopy or rolony data.

The released MATLAB matrix contains 2,167,762 cells and 114 measurement columns.
Joining the authors' 1,259,256-cell QC/type-label table by numeric cell ID gives
**1,161,387 matched cells** in 40 sections from one mouse hemisphere. The 97,869
published label IDs absent from the overlap-filtered matrix are inventoried.
The retained IDs match the matrix's non-`qc-filtered` rows exactly. The authors'
208 `Low Quality` labels are retained and flagged, not silently filtered again.

The example uses **106 named sequencing targets**, excludes five unused barcode
columns and three separate hybridization channels, and preserves raw integer
molecule counts. The paper describes a 104-gene panel, but the released matrix
and panel inventories differ. The selection and both inventories are saved; this
is an explicit adaptation, not an exact 104-gene replication claim.

There are 46 input cell-subclass categories, with missing subclass labels falling
back to published expression-derived SubCTX class labels, and 135 cluster-level
composition categories. Anatomical CCF labels never enter model training.

## Neighborhoods and training

- Coordinates: authors' corrected `fixedpos` values, in physical micrometers.
- Context: a **310 × 310 µm square**, within the same section, excluding the center.
- Input: `log2(raw molecule count + 1)`, float32; no library normalization or HVGs.
- Native model: 106 expression channels; 46 cell types; width 384; four encoder
  and four decoder layers; eight attention heads; FFN dimension 384; additive
  expression/type embedding; diagonal normalization; ZINB output; dropout 0.
- Random initialization; all approximately 8.1 million parameters trainable.
- Exactly **10 complete epochs**, retaining the final short batch. Maximum batch
  size 256, token budget 16,384, accumulation 1, seed 42 and bfloat16 mixed precision.
- Native optimizer/schedule: learning rate 0.0003; weight decay 1e-8; warmup 500
  updates and timescale 5,000. Deterministic training is enabled.
- A 95/5 within-section split yields **1,103,095 training centers** and **58,059
  validation centers**. Another 233 retained cells lack usable spatial context.
- Validation centers can occur in training neighborhoods. This measures held-out
  center prediction with shared context, not independent-section or mouse testing.
- DataLoader workers are 0, matching the demonstrated run. The low-worker warning
  is advisory; the native CPU math thread count is configured separately as 8.

The dataset adapter caches neighborhoods and calls the native `BudgetBatcher`,
model, Lightning training module, ZINB loss, optimizer and scheduler. The native
loader is checked against cached neighborhoods and expression/type tensors before
training. The production path never substitutes the CPU test's toy model.

The final completed epoch checkpoint supplies **384-dimensional attention-pool
embeddings**, exported with float32 forward execution. In the recorded run,
49,287 optimizer updates produced 1,161,154 embeddings. The selected checkpoint
is the final epoch, not the epoch with the smallest validation loss.

## Spatial domains and reference

Unsmoothed embeddings and same-section Gaussian-smoothed embeddings are each
clustered using the native cuML KMeans fitting function, seed 42 and `n_init=5`.
Smoothing uses sigma 100 µm and up to 500 neighbors including the center. The
native Gaussian formula is retained; bounded gathers and exact same-section
neighbor indexing are supplied by the adapter.

The primary reference is the per-cell anatomical name in the authors'
`neurons_registered.csv`, mapped by exact name to the Allen ontology. Its `index`
column is a palette index, not an Allen structure ID. A separate 25-µm CCF voxel
lookup is an audit only and does not replace the released labels (88.99% exact
agreement in the preparation audit).

All comparisons use the same **1,149,888 eligible reference-labelled cells**.
Empty and root/unresolved-whole-brain reference labels are excluded from evaluation.
CCF hierarchy depths 5 and 7 (root=1), and the finest populated author assignment,
give **k=63, 252 and 564**. These differ from the original CellTransformer paper's
Allen MERFISH partitions of 25/354/670. Internal-node labels represent disjoint
residual assignments. Non-grey-matter structures and small reference regions are
retained and disclosed.

## Figure 2-style measures

- **Neighborhood spatial similarity:** average cell-level fraction of the up to
  100 nearest same-section evaluation neighbors with the same label; self excluded.
- **Proportion of discrete domains:** fraction of requested domains whose mean
  cell-level score is at least the reference-domain 20th percentile. Missing
  domains count as failures.
- **Max. domain CCF similarity:** average maximum Pearson correlation between
  model-domain and reference-region cell-subclass composition vectors.
- **Matched domain CCF similarity:** maximum-sum one-to-one Pearson assignment,
  reported only when complete equal-sized matching is possible.
- Region-by-domain correlation matrices and row-normalized cell-cluster
  composition matrices provide the Figure 2g–i analogues.

Published expression-derived types are both model inputs and composition-metric
categories. Those correlations are therefore not independent validation of input
annotations. Results are exploratory anatomy comparisons from one animal; no
claim of reproduction of all original-paper experiments is made.

## Display

Reference labels receive unique categorical colors, including each cortical layer.
No reference labels or metric calculations change with the palette. Both CT2
methods retain their original domain colors. The full report includes every
section and cortical enlargements focused on motor/somatosensory cortex when
represented. Approximate outlines come from nearest-cell labels on a 20-µm raster,
clipped to within 60 µm of an observed labelled cell. They are display aids, not
the original atlas polygons and not inputs to any metric.
