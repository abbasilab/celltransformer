# Optional HTML report: Step 13

Both notebooks can generate an HTML report from the completed outputs of Steps 9–11. The report includes training curves, maps for every section and clustering resolution, and reference–domain correspondence heatmaps when reference annotations are supplied. It uses the existing domain assignments and does not retrain, extract, smooth or cluster again.

Notebook support and reporting are part of the installed `celltransformer.notebooks` package. Either notebook can be copied to its own working folder and run without companion Python files. The environment already used by the notebook supplies NumPy, pandas, SciPy, Matplotlib, PyArrow and W&B. No additional environment, GPU computation, W&B account or unattended-run manifest is needed for reporting.

For an existing installation, pull the updated repository and run `python -m pip install --no-deps -e .` from its root in the notebook's Python environment. Restart the kernel after updating. The notebooks then import `helper` from `celltransformer.notebooks` and reporting functions from `celltransformer.notebooks.reporting`. A regular wheel installation includes the same modules; an editable install is not required to run the notebook from another folder.

## Run the added step

After the earlier steps have completed, run the settings and generation cells under Step 13. The settings reuse the paths, embedding tags and k values already defined in the notebook. They do not modify those variables or the preceding analysis.

- Notebook 01 compares the full Allen example with the supplied CCF divisions, structures and substructures at the notebook's k values. Its optional cortical crops use motor/somatosensory labels, falling back to Isocortex when needed.
- Notebook 03 uses its existing k values, 20 and 50, and the example's substructure reference. For another dataset, change `REPORT_NAME`, `REPORT_COORD_UNITS`, `REFERENCE_FILE`, `REFERENCE_ID_COLUMN` and `REFERENCE_COLUMNS` in Step 13. Set `REFERENCE_COLUMNS = {}` when no reference is available.

The output directory is `WORK_DIR/html_reports/<timestamp>-<pid>/`. Each execution creates a new report, preserving earlier reports and analysis inputs. `WORK_DIR/LATEST_HTML_REPORT` contains the newest successful report directory.

The step prints paths to:

- `review.html`: the self-contained report; this single file can be copied to another computer and opened offline.
- `results.zip`: the HTML, figures, correspondence tables, matrix ordering, color dictionaries and provenance.

The notebook displays lightweight file links. If the output is outside the directory served by Jupyter, use the printed path to copy the HTML to your local computer. The report itself is not embedded into the notebook, so generating it does not add a large HTML blob to notebook outputs.

## References and interpretation

`REFERENCE_COLUMNS` maps a reference display name to an annotation column, for example `{"Histological regions": "region"}`. Reference rows are joined by exact cell ID. Row order in the reference table does not need to match the expression or embedding files. Duplicate reference cell IDs are rejected; missing labels remain unassigned. Literal category names such as `NA` are preserved when reading CSV files.

With one reference, `REFERENCE_FOR_K = None` compares it with every k in `KS`. With multiple reference levels, set a mapping such as `{20: "Broad regions", 100: "Fine regions"}`. The number of reference regions need not equal k. Neither helper assumes a fixed number of sections, anatomical labels or domains.

Each heatmap entry is the Pearson correlation between a reference region's cell-type proportions and a CT2 domain's cell-type proportions. The scale is fixed at −1 to +1. Rows and columns are ordered independently by average-linkage clustering of those proportions using Euclidean distance. Undefined correlations, including those involving empty domains, remain grey. This comparison measures similarity of cell-type composition; it is not a spatial-overlap score or a one-to-one anatomical match. No separate composition-matrix panels are displayed.

Cell types are model inputs and also enter this correspondence calculation. Validation sections were withheld during model training, but the notebooks subsequently cluster all sections jointly. The report therefore describes exploratory all-section correspondence, not an independent held-out clustering test.

Cell and section order follow the native clustering command: sorted embedding filenames, then the original row order within each section. Metadata and neighborhood-validity masks are checked across the embedding variants. Cells without valid neighborhoods are grey in domain maps and excluded from correspondence calculations; their previous influence on smoothing and clustering is preserved because neither stage is refitted.

Training curves use the selected checkpoint's local offline W&B file. The reader works on a temporary copy and does not contact W&B. It uses `train/nll_epoch` and `valid/nll_epoch`, never individual batch losses. Missing history is disclosed in the report. The output records the checkpoint, run identity, validation sections and reporting software versions.

## Optional enlargements

`REPORT_CROP = None` shows whole sections only. For an annotation-based crop, supply a dictionary with `column`, optional `prefixes` or exact `values`, `min_cells` and `padding`. The crop column must be included in the loaded reference table. Optional `fallback_column` and `fallback_values` select a larger area when too few cells match the preferred labels. Padding and boundary distances use the same units as the coordinates.

The underlying `report_from_notebook` function also accepts explicit `crop_bounds={section: [[xmin, ymin], [xmax, ymax]]}`. Supply either those bounds or `crop_selection`, not both. `boundary_grid_size` and `boundary_max_distance` enable outlines inferred from nearby reference-labelled cells. These are display approximations, not original atlas polygons.
