"""Focused tests for joins, controls, CCF identities and checksum failures."""
import json
import hashlib
import h5py
import numpy as np
import pandas as pd
import pytest
from prepare_allen import align_metadata, select_genes, references
from reference_colors import distinct_references
from download import fetch


def test_join_is_by_id_and_preserves_count_order():
    meta=pd.DataFrame({'cell_label':['c','missing','a'],'subclass':['C','X','A'],
                       'brain_section_label':['S1','S1','S1']})
    rows,selected,missing=align_metadata(np.array(['b','a','c']),meta)
    assert rows.tolist()==[1,2]
    assert selected.subclass.tolist()==['A','C']
    assert missing.cell_label.tolist()==['missing']
    with pytest.raises(ValueError,match='Duplicate annotation'):
        align_metadata(np.array(['a']),pd.concat([meta,meta.iloc[:1]]))


def test_gene_selection_preserves_order_and_removes_controls(tmp_path):
    with h5py.File(tmp_path/'genes.h5','w') as f:
        v=f.create_group('var');v.attrs['_index']='id'
        v.create_dataset('id',data=np.array(['ENSMUSG2','Blank1','ENSMUSG1'],dtype=h5py.string_dtype()))
        v.create_dataset('gene_symbol',data=np.array(['G2','Blank1','G1'],dtype=h5py.string_dtype()))
        ids,symbols,keep=select_genes(f)
        assert ids[keep].tolist()==['ENSMUSG2','ENSMUSG1']
        assert symbols[keep].tolist()==['G2','G1']


def test_residual_labels_are_retained_and_layers_have_distinct_colors(tmp_path):
    meta=pd.DataFrame({'parcellation_division':['Isocortex']*3+['unassigned'],
        'parcellation_structure':['MOp']*3+['unassigned'],
        'parcellation_substructure':['MOp1','MOp2/3','MOp5','unassigned']})
    references(meta,tmp_path)
    refs=distinct_references(json.loads((tmp_path/'references.json').read_text()))
    assert refs['substructure']['k']==4
    assert len(set(refs['substructure']['colors']))==4
    assert refs['division']['colors'][refs['division']['names'].index('unassigned')]==refs['substructure']['colors'][refs['substructure']['names'].index('unassigned')]


def test_checksum_failure_never_overwrites_existing_data(tmp_path):
    p=tmp_path/'file';p.write_bytes(b'bad')
    spec=dict(name='file',url='https://example.invalid/data',size=3,md5=hashlib.md5(b'new').hexdigest())
    with pytest.raises(ValueError,match='not overwritten'):fetch(spec,p)
    assert p.read_bytes()==b'bad'


def test_verified_existing_download_needs_no_network(tmp_path):
    p=tmp_path/'file';p.write_bytes(b'abc')
    spec=dict(name='file',url='https://example.invalid/data',size=3,md5=hashlib.md5(b'abc').hexdigest())
    fetch(spec,p);fetch(spec,p)
    assert json.loads((tmp_path/'file.verified.json').read_text())['spec']==spec
