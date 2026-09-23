# Validation performed 2026-09-21

This is a prepared, tested launcher for cactus. **A completed 10-epoch CUDA training run is not claimed.** No SSH connection to cactus or new training on its GPUs was performed while creating this package.

## Real data checks performed locally

- Downloaded all four pinned public files in full without credentials: 826,716,670 bytes total. Checked the Mendeley publisher SHA256 values for the three deposited files; pinned the observed SHA256 of the official Allen annotation volume.
- Loaded the actual sparse MATLAB matrix using SciPy; no MATLAB installation needed.
- Joined the actual published type labels by numeric cell ID, validating section prefixes and matrix IDs. Exactly 1,161,387 IDs matched, identical to the matrix's non-`qc-filtered` rows. Inventoried 97,869 published QC-label IDs without released count rows.
- Joined the full registration CSV independently by cell ID. Verified section, physical coordinates, and CCF coordinates against the matrix. Palette indices were not mistaken for CCF IDs. All nonempty/non-root region names map exactly into Allen's ontology.
- Verified all selected target values are finite, nonnegative integers and below int32 capacity. Selected the 106 named sequencing columns; excluded five unused barcodes and three separate hybridization-marker channels. Retained the conflicting author panel inventories for provenance.
- Executed preparation and graph construction on all 1,161,387 retained cells. Created 83,621,664 neighbor edges; eligible centers have median 70 neighbors, 99th percentile 144 and maximum 204 within the 310-µm square. Preparation completed successfully.
- Counted 1,161,154 eligible centers, 1,103,095 training centers, 58,059 validation centers and 1,149,888 common reference-labelled evaluation centers. Actual comparison k values are 63, 252 and 564.
- Checked 40 real centers distributed through the cohort against full-section geometric membership and the supplied native CT2 loader. Every expression/type tensor was bitwise identical.
- Enumerated the complete first-epoch batch packing: 4,929 batches, sizes 82–245, median 224. This uses maximum batch size 256 and maximum token count 16,384.
- Actual prepared data/metadata: approximately 0.89 GiB. The local preparation test used an 8-GiB free-space reserve because the Mac has limited space; production config retains a 20-GiB reserve and requires 80 GiB before setup. No data, patch-size or model-input settings were reduced in this full-cohort test.
- Generated and visually inspected actual registered anatomical cell maps and cortical enlargements. `reference_preview.png` contains only public reference labels, with no fabricated model outcomes.

## Software tests passed

- `test_barseq.py`: shuffled ID joins, missing-ID inventory, duplicate IDs, wrong sections, QC-mask disagreement, missing-type fallback, explicit gene selection despite changed column order, axis-specific CCF lookup, invalid coordinates, and same/different-label display boundaries.
- `test_adapter.py`: native loader tensor equality for all 96 toy cells; 12 batch-size/token-budget combinations with exact center coverage; native Lightning loss/optimizer/scheduler called through an explicitly labelled toy forward model for CPU orchestration tests.
- Checkpoint recovery: complete one epoch, intentionally interrupt the next, restore the completed-epoch checkpoint, then verify exact epoch coverage and uninterrupted update-number history after rollback of partial logs.
- Gaussian smoothing: unchanged native weights, independent two-point analytical values, no cross-section mixing.
- Composition: independent permutation matching example with maximum and one-to-one Pearson correlations of 1. Undefined/incomplete matching is represented explicitly.
- Offline report and ZIP generated from synthetic tests. Embedded PNGs and selector inventories checked; scientific plots inspected as PNG files. Browser preview of a local file URL was blocked by browser security policy, so browser interaction/rendering is not certified.
- Python compilation, Bash syntax checks and source-manifest comparison. All 66 bundled CT2 files match the supplied repository byte for byte.

## Checks enforced on cactus before training

1. Fresh standalone environment setup and PyTorch CUDA/bfloat16 support.
2. GPU KMeans test using the actual unchanged native CT2 fitting function in the RAPIDS environment.
3. Exact native model construction, batch isolation, masked-center nonleakage, full backward pass with finite nonzero gradients in core components, and a dense-neighborhood batch within the token budget.
4. Finite losses during training; every training center exactly once per epoch; finite exported embeddings; complete populated KMeans domains; stage output checks before SUCCESS.

The native model, CUDA kernels, GPU memory use, and fresh Linux dependency solver have not been executed locally on this Mac. The launcher therefore stops on a failed cactus preflight instead of silently using a different model or CPU fallback. Timing and the 50–70 GiB total installation/run budget remain estimates until measured on cactus.

The synthetic tests exercise software behavior only. Their reports are not evidence of biological performance. Ten epochs are the requested proof-of-concept budget, not an assertion of convergence or optimal anatomical agreement.
