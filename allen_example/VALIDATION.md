# Validation record

Checked locally on 24 September 2026. These are software and input-data checks;
they are not a completed Allen GPU training experiment.

## Real public data

- Fetched the official 20231215 manifest and pinned all nine selected files.
- Downloaded the full 1.61-GB combined cell/type/CCF annotation table and verified
  its published MD5: `51dda47ab139e91357bc09bc4e12c073`.
- Verified the gene table, taxonomy table, parcellation terms, memberships,
  acronym table and two compressed atlas volumes against their published MD5s.
- Inspected the raw HDF5 using HTTP byte ranges: CSR counts, 4,334,174 rows and
  550 channels. Read its complete cell-ID index and verified that all 3,739,961
  annotation rows occur in the count matrix. Fifty channels are Blank controls.
- Audited the full annotation coordinate table with the specified 85-µm square
  neighborhoods: 3,737,501 eligible centers, 2,460 isolated cells and 73,834,240
  directed context edges. Neighbor counts at eligible centers have minimum 1,
  median 17, 90th percentile 35, 99th percentile 63 and maximum 217.
- All three reference partitions remain populated on eligible centers:
  25 divisions, 354 structures and 670 substructures.
- Fetched a 4,096-row measured-count fixture in section 37. Of these, 3,194 cells
  have released annotations. Shuffled annotation order before preparation, then
  confirmed that all selected count rows, genes and anatomical labels matched
  the independently indexed originals. All 500 measured genes were retained.
- Compared cached neighborhoods against the native CT2 sampler at 20 eligible
  centers in that real fixture: expression and type tensors were bitwise equal.
  This small fixture does not represent a complete section or a biological fit.

The full 7.63-GB raw expression file was **not** downloaded locally. The remote
workflow verifies its full published checksum before preparing or training.

## Software checks

- **41 original CPU tests passed:** distributions, loaders and preprocessing.
- **Five Allen data tests passed:** cell-ID order and duplicate rejection,
  Blank-control exclusion, exact residual-region labels and unique layer
  colors, non-destructive checksum failure, and verified-file reuse.
- **Coordinator recovery test passed:** source snapshot, failed training,
  resumption without repeating preparation, stale-error removal and the final
  success marker. Stage execution was mocked for this test.
- Synthetic CPU adapter test passed: exact center coverage under 12 batching
  configurations, native loss/optimizer/scheduler with an explicitly test-only
  toy forward model, interruption mid-epoch, completed-checkpoint restoration,
  and consistent resumed history.
- Analytical Gaussian-weight check and absence of cross-section mixing passed.
- Composition matching under a known permutation returned correlation 1.
- Standalone synthetic HTML and ZIP generation passed; every test section and
  resolution was embedded. A generated section PNG was visually inspected.
- All 66 supplied CT2 files match their source SHA256 manifest.
- Python compilation and shell syntax checks passed.

## Checks performed on the GPU workstation

The workflow runs dependency checks, a real GPU KMeans preflight, cached/native
sampler equivalence checks, center-masking and batch-isolation checks, and a
bfloat16 forward/backward test on dense neighborhoods before the full fit.
The full model, training duration and resulting anatomical metrics must still
be verified there. No CPU test output is a substitute for these checks.

The package supports reuse of the previously demonstrated BARseq environments.
Installing new environments uses the same package pins but has not been tested
on a second Linux GPU workstation during this task.

## Integration with the two-example repository

Checked 25 September 2026 with distribution version 0.2.0:

- Both adapters use the same unchanged 66-file native source snapshot.
- The source archive builds a wheel containing both Allen and BARseq runtime
  resources. Package metadata validation and installed dependency checks pass.
- All eight installed command help screens pass, including native Hydra training
  configuration resolution outside the checkout.
- Both examples independently generate synthetic HTML and ZIP reports from the
  installed wheel, including completed-checkpoint recovery and metric checks.
- Editable installation resolves both adapters from the checkout and passes both
  source audits. CLI tests cover launch, resume, status, workflow identity and
  Allen environment-reuse arguments.

These installation tests do not rerun either full biological GPU experiment.
