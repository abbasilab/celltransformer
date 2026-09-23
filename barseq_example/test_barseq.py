"""Independent failure-case tests for release joins and atlas lookups."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_barseq import select_cells, annotate_types, lookup_atlas, select_genes
from report import reference_boundaries
from common import read_json,HERE

m=dict(id=np.array([11,22,33]),slice=np.array([1,2,2]),clustid=np.array(['A','qc-filtered','non_Exc']))
labels=pd.DataFrame(dict(sample=['2_99','2_33','1_11'],**{'class':['X','SubCTX_1','Glutamatergic'],'subclass':['NA','NA','L6 IT'],'cluster':['NA','NA','L6a']}))
keep,selected,absent=select_cells(m,labels)
assert keep.tolist()==[True,False,True] and selected['sample'].tolist()==['1_11','2_33'] and absent['sample'].tolist()==['2_99']
typed=annotate_types(selected);assert typed.cell_subclass.tolist()==['L6 IT','SubCTX_1'] and typed.cell_cluster.tolist()==['L6a','SubCTX_1']
for broken in [pd.concat([labels,labels.iloc[[0]]]), labels.assign(sample=['2_99','1_33','1_11']),labels.iloc[:2]]:
    try:select_cells(m,broken)
    except ValueError:pass
    else:raise AssertionError('Bad join was accepted')
genes=read_json(HERE/'reference/sequencing_genes.json');names=['unused-1']+genes[::-1]+['Slc17a7','Gad1','Slc30a3'];_,chosen,index=select_genes(dict(genes=np.c_[names,names]));assert [names[i] for i in index]==genes and len(chosen)==106
# Axis order, zero-based origin, rounding, and invalid coordinates tested separately.
vol=np.arange(2*3*4).reshape((2,3,4));points=np.array([[1.,2.,3.],[0,1,2],[.49,1.51,1.49],[-1,0,0],[2,0,0],[np.nan,1,1]])
assert lookup_atlas(points,vol).tolist()==[23,6,9,0,0,0]
# No boundary between same labels; one vertical boundary between two labels.
xy=np.array([[0.,0.],[40.,0.]]);bounds=(np.array([0.,0.]),np.array([40.,0.]))
assert len(reference_boundaries(xy,np.array([0,0]),bounds))==0
edges=reference_boundaries(xy,np.array([0,1]),bounds);assert len(edges)==1
print('BARSEQ RELEASE JOIN, PANEL, CCF AXES AND DISPLAY BOUNDARY TESTS: PASSED')

# A fully downloaded .partial file is verified and finalized on resume, without
# issuing a Range request beyond EOF. A corrupt completed partial is preserved.
import hashlib
from prepare_barseq import fetch
with tempfile.TemporaryDirectory() as folder:
    dest=Path(folder)/'counts.mat';payload=b'complete interrupted transfer';spec=dict(url='https://invalid.example/not-requested',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    partial=dest.with_suffix('.mat.partial');partial.write_bytes(payload);fetch(spec,dest)
    assert dest.read_bytes()==payload and not partial.exists()
    fetch(spec,dest)
    bad=Path(folder)/'bad.mat';bad.with_suffix('.mat.partial').write_bytes(b'x'*len(payload))
    try:fetch(spec,bad)
    except ValueError:pass
    else:raise AssertionError('Corrupt completed partial accepted')
    assert not bad.exists() and bad.with_suffix('.mat.partial').exists()
print('DOWNLOAD COMPLETION/INTEGRITY RESUME TESTS: PASSED')
