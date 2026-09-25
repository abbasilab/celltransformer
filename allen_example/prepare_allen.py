"""Allen 1, frozen 20231215 taxonomy/CCF; exact cell-ID joins, measured genes only."""
import argparse
import shutil
import h5py
import pandas as pd
from scipy.spatial import cKDTree
from common import *

EXPECTED = dict(raw_cells=4334174, raw_channels=550, annotated_cells=3739961,
                cells=3739961, sections=53, genes=500, subclasses=338)
LEVELS = {'division':25, 'structure':354, 'substructure':670}
COLUMNS = ['cell_label','brain_section_label','class','subclass','cluster',
           'x_section','y_section','x_reconstructed','y_reconstructed',
           'parcellation_index','parcellation_category','parcellation_division',
           'parcellation_structure','parcellation_substructure']


def load_metadata(path):
    categorical_cols=[c for c in COLUMNS if c not in ['cell_label','x_section','y_section',
                     'x_reconstructed','y_reconstructed','parcellation_index']]
    return pd.read_csv(path,usecols=COLUMNS,keep_default_na=False,
                       dtype={**{c:'category' for c in categorical_cols},'cell_label':str})


def select_genes(f):
    var=f['var'];key=var.attrs['_index'];ids=var[key].asstr()[:]
    require(len(set(ids))==len(ids),'Duplicate gene identifiers')
    if 'gene_symbol' in var and isinstance(var['gene_symbol'],h5py.Dataset):
        symbols=var['gene_symbol'].asstr()[:]
    elif 'gene_symbol' in var:
        codes,cats=categorical(var['gene_symbol']);symbols=np.array(cats)[codes]
    else: raise ValueError('Expected released gene_symbol column')
    keep=np.array([not str(i).startswith('Blank') and not str(s).startswith('Blank') for i,s in zip(ids,symbols)])
    require(all(str(i).startswith('ENSMUSG') for i in ids[keep]),'Unexpected non-gene channel')
    return ids,symbols,keep


def align_metadata(raw_ids,meta):
    require(not pd.Index(raw_ids).has_duplicates,'Duplicate count cell IDs')
    require(not meta.cell_label.duplicated().any(),'Duplicate annotation cell IDs')
    lookup=pd.Index(meta.cell_label).get_indexer(raw_ids)
    rows=np.flatnonzero(lookup>=0)
    # Preserve raw matrix order, and save the mapping for every exported row.
    selected=meta.iloc[lookup[rows]].copy().reset_index(drop=True)
    require(np.array_equal(selected.cell_label.to_numpy(),raw_ids[rows]),'Cell ID alignment failed')
    missing=meta.loc[~meta.cell_label.isin(raw_ids),['cell_label','brain_section_label']]
    return rows,selected,missing


def references(meta,out):
    refs={};vocab={}
    for level in LEVELS:
        column='parcellation_'+level
        require(not meta[column].isna().any() and not (meta[column]=='').any(),'Missing released anatomy label')
        names=sorted(meta[column].unique().tolist())
        values=pd.Categorical(meta[column],categories=names).codes.astype(np.int32)
        atomic_npy(out/f'reference_{level}.npy',values)
        refs[level]=dict(k=len(names),ids=names,names=names,colors=['#808080']*len(names),
            label='CCF '+level,definition='Exact 20231215 released parcellation_'+level+' labels, including explicit unassigned/residual categories; no ontology-depth substitution')
        vocab['reference_'+level]=names
    json_write(out/'references.json',refs)
    return vocab


def graph_and_split(run):
    cfg=configuration(run);out=Path(run)/'prepared';xy=array(run,'xy_um');sec=array(run,'sections');nonempty=array(run,'nonempty');n=len(xy)
    degree=np.zeros(n,np.int64);half=cfg['patch_width_um']/2
    for s in np.unique(sec):
        ids=np.flatnonzero(sec==s);tree=cKDTree(xy[ids])
        for a,b in chunks(len(ids),4096):degree[ids[a:b]]=tree.query_ball_point(xy[ids[a:b]],half,p=np.inf,return_length=True,workers=cfg['threads'])-1
    ptr=np.r_[0,np.cumsum(degree)];require((degree>=0).all(),'Invalid degree')
    needed=int(ptr[-1])*4+n*384*8+int(cfg.get('storage_reserve_gib',20))*1024**3
    require(shutil.disk_usage(out).free>needed,f'Graph and two embedding arrays need about {needed/1024**3:.1f} GiB additional free space')
    atomic_npy(out/'neighbor_indptr.npy',ptr);edges=mmap_create(out/'neighbor_indices.npy',(int(ptr[-1]),),np.int32)
    for s in np.unique(sec):
        ids=np.flatnonzero(sec==s);tree=cKDTree(xy[ids]);order=np.argsort(xy[ids,0],kind='stable');rank=np.empty(len(ids),np.int64);rank[order]=np.arange(len(ids))
        for a,b in chunks(len(ids),2048):
            near=tree.query_ball_point(xy[ids[a:b]],half,p=np.inf,workers=cfg['threads'])
            for i,local in zip(ids[a:b],near):
                local=np.asarray(local,np.int64);local=local[ids[local]!=i];local=local[np.argsort(rank[local],kind='stable')];edges[ptr[i]:ptr[i+1]]=ids[local]
        log(f'Cached CT2 neighborhoods for section {s+1}/{len(np.unique(sec))}')
    edges.flush();eligible=degree>0;atomic_npy(out/'eligible.npy',eligible)
    rng=np.random.default_rng(cfg['seed']);ids=np.flatnonzero(eligible)
    require(len(ids)>1,'Fewer than two eligible centers after the cell-ID join/neighborhood construction')
    val=np.sort(rng.choice(ids,max(1,round(len(ids)*cfg['validation_fraction'])),replace=False)).astype(np.int64)
    mask=eligible.copy();mask[val]=False;train=np.flatnonzero(mask)
    require(len(train)>0 and len(val)>0,'Empty split')
    atomic_npy(out/'train_ids.npy',train);atomic_npy(out/'validation_ids.npy',val)
    return dict(n_eligible=int(eligible.sum()),n_train=len(train),n_validation=len(val),n_edges=int(ptr[-1]),neighbor_quantiles=np.quantile(degree[eligible],[0,.5,.9,.99,1]).tolist())



def prepare(run):
    run=Path(run);cfg=configuration(run);raw=Path(cfg['root'])/'data/raw';out=run/'prepared';out.mkdir(exist_ok=True)
    production=not cfg.get('software_test',False)
    log('Reading pinned cell-type and CCF annotations')
    meta=load_metadata(raw/'cells_ccf.csv');n_annotated=len(meta)
    with h5py.File(raw/'expression_raw.h5ad','r') as f:
        x=f['X'];require(x.attrs.get('encoding-type')=='csr_matrix','Expected released CSR expression matrix')
        shape=tuple(int(v) for v in x.attrs['shape']);raw_ids=f['obs'][f['obs'].attrs['_index']].asstr()[:]
        genes,symbols,chosen=select_genes(f)
        rows,meta,missing=align_metadata(raw_ids,meta)
        require(len(rows)>1,'The count file and annotated-cell table have fewer than two shared cell IDs')
        if production:
            require(shape==(EXPECTED['raw_cells'],EXPECTED['raw_channels']),'Unexpected raw matrix inventory')
            require(n_annotated==EXPECTED['annotated_cells'],'Unexpected annotation row count')
            require(len(rows)==EXPECTED['cells'] and int(chosen.sum())==EXPECTED['genes'],
                    f'Pinned release cohort/gene inventory mismatch: {len(rows)} cells, {chosen.sum()} genes')
            require(meta.brain_section_label.nunique()==EXPECTED['sections'],'Unexpected section count')
            require(meta.subclass.nunique()==EXPECTED['subclasses'],'Unexpected subclass count')
        missing.to_csv(out/'annotation_ids_without_counts.tsv',sep='\t',index=False)
        atomic_npy(out/'raw_row_indices.npy',rows)
        pd.DataFrame(dict(gene_identifier=genes,gene_symbol=symbols,selected=chosen,
                         reason=np.where(chosen,'Measured biological gene','Blank control probe'))).to_csv(out/'gene_selection.tsv',sep='\t',index=False)
        json_write(out/'genes.json',genes[chosen].tolist());json_write(out/'gene_symbols.json',symbols[chosen].tolist())
        n=len(rows);target=mmap_create(out/'counts.npy',(n,int(chosen.sum())),np.int32);total=np.zeros(n,np.int64);ngenes=np.zeros(n,np.int32)
        # Consecutive CSR reads are fast and do not load the full sparse matrix.
        for a,b in chunks(shape[0],8192):
            lo,hi=np.searchsorted(rows,[a,b]);local=rows[lo:hi]-a
            if not len(local):continue
            block=csr_block(x,a,b)[local][:,chosen].toarray()
            require(np.isfinite(block).all() and (block>=0).all() and np.equal(block,np.floor(block)).all(),
                    'Raw molecule counts must be finite, nonnegative integers; no rounding is applied')
            require(block.max(initial=0)<=np.iinfo(np.int32).max,'Count overflow')
            target[lo:hi]=block.astype(np.int32);total[lo:hi]=block.sum(1,dtype=np.int64);ngenes[lo:hi]=(block>0).sum(1)
            if a//262144 != b//262144:log(f'Counts: scanned {b:,}/{shape[0]:,} raw cells')
        target.flush()
    meta=meta.rename(columns={'cell_label':'cell_id','brain_section_label':'section','class':'cell_class',
                             'subclass':'cell_subclass','cluster':'cell_cluster'})
    xy=meta[['x_reconstructed','y_reconstructed']].to_numpy(np.float64)*1000.
    require(np.isfinite(xy).all(),'Nonfinite physical coordinates')
    meta['x_um']=xy[:,0];meta['y_um']=xy[:,1];meta['total_counts']=total;meta['n_genes']=ngenes
    meta['reference_valid']=True
    vocab=references(meta,out)
    for column,name in [('section','section'),('cell_subclass','cell_subclass'),('cell_cluster','cell_cluster'),('cell_class','cell_class')]:
        cats=sorted(meta[column].unique().tolist());require('' not in cats,'Empty cell type annotation')
        vocab[name]=cats
        code=pd.Categorical(meta[column],categories=cats).codes.astype(np.int32)
        atomic_npy(out/({'section':'sections','cell_subclass':'types'}.get(name,name)+'.npy'),code)
    require(len(vocab['cell_subclass'])<=cfg['model']['cell_type_cardinality'],'Cell types exceed model capacity')
    json_write(out/'vocabulary.json',vocab);atomic_npy(out/'xy_um.npy',xy);atomic_npy(out/'nonempty.npy',total>0)
    meta.to_parquet(out/'cells.parquet',index=False)
    graph=graph_and_split(run)
    eligible=array(run,'eligible')
    for level,k in LEVELS.items():
        observed=len(np.unique(array(run,'reference_'+level)[eligible]))
        if production:require(observed==k,f'Expected {k} populated {level} labels on eligible centers, found {observed}')
    inventory=meta.groupby('section',observed=True).agg(cells=('cell_id','size'),total_counts=('total_counts','sum'))
    inventory.to_csv(out/'section_inventory.tsv',sep='\t')
    summary=dict(paper_reported_cells=3737550, released_annotation_cells=3739961, paper_count_difference_note='Released annotation/matrix intersection has 2,411 more rows than the reported paper total. Exact original analysis cell list is not established by these release files; all verified rows retained and eligible count reported separately.',dataset='Allen 1 / MERFISH-C57BL6J-638850',release='20231215',raw_matrix_shape=shape,
        n_annotation_rows=n_annotated,n_annotation_ids_without_counts=len(missing),n_cells=n,n_sections=len(vocab['section']),
        n_model_genes=int(chosen.sum()),n_model_cell_types=len(vocab['cell_subclass']),n_mice=1,
        n_zero_count_cells=int((total==0).sum()),n_excluded_without_neighbors=int((~eligible).sum()),zero_count_policy='Retained as context and centers if they have neighbors',
        **graph,coordinate_conversion='x_reconstructed/y_reconstructed in mm multiplied by 1000; CCF-warped coordinates are not used for model neighborhoods',
        gene_order='Original raw H5 order, excluding 50 Blank controls; Ensembl IDs and symbols exported',
        cell_order='Intersection of released count IDs and 20231215 annotated/registered IDs, in count-matrix order',
        reference_counts={level:len(vocab['reference_'+level]) for level in LEVELS},
        validation='Random 80/20 split of eligible centers across the dataset; spatial context overlaps; not held-out-section/animal evaluation',
        software_test=not production)
    json_write(out/'summary.json',summary);log('ALLEN 1 PREPARATION: PASSED');log(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run)
