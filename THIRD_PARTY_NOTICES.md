# Attribution and distribution scope

The original CellTransformer code is copyright (c) 2026 Alex Lee, MIT licensed;
the full original license is retained at the root. Its supplied distribution
implementation and other inline provenance comments remain intact.

The example scripts are covered by the same license. Code licensing does not relicense external data.

The Allen 1 MERFISH expression and cell/CCF annotation data are downloaded from
the Allen Brain Cell Atlas. The source collection is licensed CC BY-NC 4.0:
https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html.
The example pins the 20231215 manifest and preserves published hashes and release
paths. Large expression files, annotation volumes and cell tables are downloaded
at runtime; they are not distributed with this repository. The download manifest
`allen_example/DOWNLOAD_MANIFEST.json` identifies those resources. Cite Yao and colleagues:
https://doi.org/10.1038/s41586-023-06812-z. Preserve the source data's terms when
sharing data or derived results; the MIT code license does not replace them.

The public BARseq pilot deposit by Xiaoyin Chen and Stephan Fischer is licensed
CC BY 4.0: https://doi.org/10.17632/8bhhk7c5n9.1. The example downloads its processed
matrix, type labels and registered cell table rather than redistributing those
large inputs. The download manifest `barseq_example/DOWNLOAD_MANIFEST.json`
identifies those resources.

The example's gene list (`barseq_example/reference/sequencing_genes.json`) follows
the authors' gene panel, supplementary table and public analysis code:
https://github.com/gillislab/barseq_analysis/tree/main/pilot and
https://doi.org/10.1038/s41586-024-07221-6. That external R/MATLAB/registration code
is not required by this example and is not redistributed here.

The Allen structure ontology and annotation are from the Allen Institute for
Brain Science. The downloaded annotation volumes remain external to Git; the
small ontology is included as `barseq_example/reference/allen_structure_graph.json`, a copy of
https://api.brain-map.org/api/v2/structure_graph_download/1.json.
See https://doi.org/10.1016/j.cell.2020.04.007 and
https://alleninstitute.org/legal/terms-of-use/ for the originating reference and
data-use terms. This repository does not change the ownership of those resources.

Python, PyTorch, Lightning, RAPIDS and other dependencies retain their respective
licenses. Their binaries and installed environments are not committed to this
repository; `environment/` holds the installation definitions and version records.
