# Attribution and distribution scope

The original CellTransformer2 code is copyright (c) 2026 Alex Lee, MIT licensed;
the full original license is retained at the root. Its supplied distribution
implementation and other inline provenance comments remain intact.

The BARseq example scripts are distributed under the MIT license in
`barseq_example/LICENSE`. Code licensing does not relicense external data.

The public BARseq pilot deposit by Xiaoyin Chen and Stephan Fischer is licensed
CC BY 4.0: https://doi.org/10.17632/8bhhk7c5n9.1. The example downloads its processed
matrix, type labels and registered cell table rather than redistributing those
large inputs. Selected figures and numeric outputs in `barseq_example/results/`
are derived from that deposit; their changes comprise CellTransformer2 analysis,
figure rendering and a categorical recoloring of anatomical labels. Attribute
the data authors and the analysis when reusing them. Derived results are shared
under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.

The small gene inventories and provenance records refer to the authors' gene
panel, supplementary table and public analysis code:
https://github.com/gillislab/barseq_analysis/tree/main/pilot and
https://doi.org/10.1038/s41586-024-07221-6. That external R/MATLAB/registration code
is not required by this adapter and is not redistributed here.

The Allen structure ontology and annotation are from the Allen Institute for
Brain Science. The downloaded annotation volume remains external to Git; the
small ontology is included with its source URL in the example documentation.
See https://doi.org/10.1016/j.cell.2020.04.007 and
https://alleninstitute.org/legal/terms-of-use/ for the originating reference and
data-use terms. This repository does not change the ownership of those resources.

Python, PyTorch, Lightning, RAPIDS and other dependencies retain their respective
licenses. Their binaries and installed environments are not committed to this
repository; the example supplies installation definitions and version records.
