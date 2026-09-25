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

#load cell annotations
def load_metadata(path):
    """
    Read the per-cell annotation table (cell type labels, section,
    coordinates, and CCF parcellation) from a CSV file.
    """

    #define which columns are categorical and which are not
    non_categorical_cols = ['cell_label', 'x_section', 'y_section',
                            'x_reconstructed', 'y_reconstructed', 'parcellation_index']
    
    categorical_cols=[c for c in COLUMNS if c not in non_categorical_cols]

    #build the column -> dtype mapping for read_csv
    column_dtypes = {col: 'category' for col in categorical_cols}
    column_dtypes['cell_label'] = str  #keep cell IDs as plain strings

    #load metadata
    metadata = pd.read_csv(path, usecols=COLUMNS, keep_default_na=False, dtype=column_dtypes)
    return metadata

    #Old, claude:
    # categorical_cols=[c for c in COLUMNS if c not in ['cell_label','x_section','y_section',
    #                  'x_reconstructed','y_reconstructed','parcellation_index']]
    # return pd.read_csv(path,usecols=COLUMNS,keep_default_na=False,
    #                    dtype={**{c:'category' for c in categorical_cols},'cell_label':str})

#drop blank genes
def select_genes(f):
    """
    Read gene IDs and symbols from an h5ad file's `var` group,
    and figure out which genes are real (not QC 'blank' probes).
    """

    var = f['var']

    #get gene IDs
    index_column_name = var.attrs['_index']
    gene_ids = var[index_column_name].asstr()[:]

    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Duplicate gene identifiers found")

    #get gene symbols
    if 'gene_symbol' not in var:
        raise ValueError("Expected a gene_symbol column, but didn't find one")

    symbol_data = var['gene_symbol']
    if isinstance(symbol_data, h5py.Dataset):
        #in case it's stored as plain string array
        gene_symbols = symbol_data.asstr()[:]
    else:
        #in case it's stored as categorical (decodes codes into actual strings)
        codes, categories = categorical(symbol_data)
        gene_symbols = np.array(categories)[codes]

    #drop blank genes
    keep_mask = np.array([
        not (str(gene_id).startswith('Blank') or str(symbol).startswith('Blank'))
        for gene_id, symbol in zip(gene_ids, gene_symbols)
    ])

    #sanity check that each gene is Ensembl mouse gene ID (starts with "ENSMUSG")
    real_gene_ids = gene_ids[keep_mask]
    if not all(str(i).startswith('ENSMUSG') for i in real_gene_ids):
        raise ValueError("Found a gene ID that doesn't look like a mouse Ensembl ID")

    return gene_ids, gene_symbols, keep_mask

    #Old, claude:
    # var=f['var'];key=var.attrs['_index'];ids=var[key].asstr()[:]
    # require(len(set(ids))==len(ids),'Duplicate gene identifiers')
    # if 'gene_symbol' in var and isinstance(var['gene_symbol'],h5py.Dataset):
    #     symbols=var['gene_symbol'].asstr()[:]
    # elif 'gene_symbol' in var:
    #     codes,cats=categorical(var['gene_symbol']);symbols=np.array(cats)[codes]
    # else: raise ValueError('Expected released gene_symbol column')
    # keep=np.array([not str(i).startswith('Blank') and not str(s).startswith('Blank') for i,s in zip(ids,symbols)])
    # require(all(str(i).startswith('ENSMUSG') for i in ids[keep]),'Unexpected non-gene channel')
    # return ids,symbols,keep


def align_metadata(raw_ids, meta):
    """
    Match cells in the count matrix to their annotation rows by cell ID,
    keeping the count matrix's cell order.
    """

    #check for duplicates
    if pd.Index(raw_ids).has_duplicates:
        raise ValueError('Duplicate count cell IDs')
    if meta.cell_label.duplicated().any():
        raise ValueError('Duplicate annotation cell IDs')

    #for each count-matrix cell, find its row in meta (-1 if it has no annotation)
    meta_ids = pd.Index(meta.cell_label)
    meta_row_for_each_cell = meta_ids.get_indexer(raw_ids)

    #count-matrix rows that have an annotation
    has_annotation = meta_row_for_each_cell >= 0
    rows = np.flatnonzero(has_annotation)

    #pull the matching annotation rows, in count-matrix order
    selected = meta.iloc[meta_row_for_each_cell[rows]].copy()
    selected = selected.reset_index(drop=True)

    #sanity check that IDs line up row by row
    if not np.array_equal(selected.cell_label.to_numpy(), raw_ids[rows]):
        raise ValueError('Cell ID alignment failed')

    #annotated cells that aren't in the count matrix
    is_missing = ~meta.cell_label.isin(raw_ids)
    missing = meta.loc[is_missing, ['cell_label', 'brain_section_label']]

    return rows, selected, missing

    #Old, claude:
    # require(not pd.Index(raw_ids).has_duplicates,'Duplicate count cell IDs')
    # require(not meta.cell_label.duplicated().any(),'Duplicate annotation cell IDs')
    # lookup=pd.Index(meta.cell_label).get_indexer(raw_ids)
    # rows=np.flatnonzero(lookup>=0)
    # # Preserve raw matrix order, and save the mapping for every exported row.
    # selected=meta.iloc[lookup[rows]].copy().reset_index(drop=True)
    # require(np.array_equal(selected.cell_label.to_numpy(),raw_ids[rows]),'Cell ID alignment failed')
    # missing=meta.loc[~meta.cell_label.isin(raw_ids),['cell_label','brain_section_label']]
    # return rows,selected,missing


def references(meta, out):
    """
    For each CCF level (division, structure, substructure), save each cell's
    region as an integer code, and write a references.json describing the labels.
    """

    refs = {}
    vocab = {}

    for level in LEVELS:
        column = 'parcellation_' + level

        #check no cell is missing a region label
        has_nan = meta[column].isna().any()
        has_empty = (meta[column] == '').any()
        if has_nan or has_empty:
            raise ValueError('Missing released anatomy label')

        #sorted list of region names at this level
        names = sorted(meta[column].unique().tolist())

        #convert each cell's region name to its index in names, and save
        codes = pd.Categorical(meta[column], categories=names).codes.astype(np.int32)
        atomic_npy(out / f'reference_{level}.npy', codes)

        #info about this level for references.json
        refs[level] = dict(
            k=len(names),
            ids=names,
            names=names,
            colors=['#808080'] * len(names),  #placeholder grey
            label='CCF ' + level,
            definition='Exact 20231215 released parcellation_' + level
                       + ' labels, including explicit unassigned/residual categories;'
                       + ' no ontology-depth substitution',
        )
        vocab['reference_' + level] = names

    json_write(out / 'references.json', refs)
    return vocab

    #Old, claude:
    # refs={};vocab={}
    # for level in LEVELS:
    #     column='parcellation_'+level
    #     require(not meta[column].isna().any() and not (meta[column]=='').any(),'Missing released anatomy label')
    #     names=sorted(meta[column].unique().tolist())
    #     values=pd.Categorical(meta[column],categories=names).codes.astype(np.int32)
    #     atomic_npy(out/f'reference_{level}.npy',values)
    #     refs[level]=dict(k=len(names),ids=names,names=names,colors=['#808080']*len(names),
    #         label='CCF '+level,definition='Exact 20231215 released parcellation_'+level+' labels, including explicit unassigned/residual categories; no ontology-depth substitution')
    #     vocab['reference_'+level]=names
    # json_write(out/'references.json',refs)
    # return vocab


def graph_and_split(run):
    """
    Build each cell's neighborhood (other cells in the same section inside a
    square patch around it), save it as a CSR-style edge list, then randomly
    split cells that have neighbors into train/validation centers.
    """

    cfg = configuration(run)
    out = Path(run) / 'prepared'
    xy = array(run, 'xy_um')
    sections = array(run, 'sections')
    nonempty = array(run, 'nonempty')  #loaded but not used below
    n_cells = len(xy)
    half_width = cfg['patch_width_um'] / 2
    section_ids = np.unique(sections)

    #pass 1: count each cell's neighbors
    #p=np.inf makes the search region a square (max of |dx|, |dy|), not a circle
    degree = np.zeros(n_cells, np.int64)
    for s in section_ids:
        cells_in_section = np.flatnonzero(sections == s)
        tree = cKDTree(xy[cells_in_section])
        for a, b in chunks(len(cells_in_section), 4096):
            chunk_cells = cells_in_section[a:b]
            n_in_patch = tree.query_ball_point(xy[chunk_cells], half_width, p=np.inf,
                                               return_length=True, workers=cfg['threads'])
            degree[chunk_cells] = n_in_patch - 1  #-1 so a cell doesn't count itself

    #cell i's neighbors go in edges[ptr[i]:ptr[i+1]]
    ptr = np.r_[0, np.cumsum(degree)]
    if (degree < 0).any():
        raise ValueError('Invalid degree')
    n_edges = int(ptr[-1])

    #check disk space for the edge list plus two embedding arrays
    reserve_gib = int(cfg.get('storage_reserve_gib', 20))
    needed = n_edges * 4 + n_cells * 384 * 8 + reserve_gib * 1024**3
    if shutil.disk_usage(out).free <= needed:
        raise ValueError(f'Graph and two embedding arrays need about {needed/1024**3:.1f} GiB additional free space')

    atomic_npy(out / 'neighbor_indptr.npy', ptr)
    edges = mmap_create(out / 'neighbor_indices.npy', (n_edges,), np.int32)

    #pass 2: fill in each cell's neighbor list, sorted by x coordinate
    #TODO(speed): this rebuilds the same KD-tree per section that pass 1 already built.
    #reusing it would be faster, but means either keeping every section's tree in memory
    #between passes, or merging both passes into one loop (harder to read). worth it?
    for s in section_ids:
        cells_in_section = np.flatnonzero(sections == s)
        tree = cKDTree(xy[cells_in_section])

        #rank of each cell within the section when sorted by x
        order = np.argsort(xy[cells_in_section, 0], kind='stable')
        x_rank = np.empty(len(cells_in_section), np.int64)
        x_rank[order] = np.arange(len(cells_in_section))

        for a, b in chunks(len(cells_in_section), 2048):
            chunk_cells = cells_in_section[a:b]
            neighbors_per_cell = tree.query_ball_point(xy[chunk_cells], half_width, p=np.inf,
                                                       workers=cfg['threads'])
            for cell, local in zip(chunk_cells, neighbors_per_cell):
                #local = neighbor positions within this section
                local = np.asarray(local, np.int64)
                local = local[cells_in_section[local] != cell]  #drop the cell itself
                local = local[np.argsort(x_rank[local], kind='stable')]  #sort by x
                edges[ptr[cell]:ptr[cell + 1]] = cells_in_section[local]  #store as global cell index
        log(f'Cached CT2 neighborhoods for section {s+1}/{len(section_ids)}')
    edges.flush()

    #cells with at least one neighbor can be patch centers
    eligible = degree > 0
    atomic_npy(out / 'eligible.npy', eligible)

    #random train/validation split of eligible centers
    rng = np.random.default_rng(cfg['seed'])
    eligible_ids = np.flatnonzero(eligible)
    if len(eligible_ids) < 2:
        raise ValueError('Fewer than two eligible centers after the cell-ID join/neighborhood construction')

    n_val = max(1, round(len(eligible_ids) * cfg['validation_fraction']))
    val = rng.choice(eligible_ids, n_val, replace=False)
    val = np.sort(val).astype(np.int64)

    train_mask = eligible.copy()
    train_mask[val] = False
    train = np.flatnonzero(train_mask)
    if len(train) == 0 or len(val) == 0:
        raise ValueError('Empty split')

    atomic_npy(out / 'train_ids.npy', train)
    atomic_npy(out / 'validation_ids.npy', val)

    return dict(
        n_eligible=int(eligible.sum()),
        n_train=len(train),
        n_validation=len(val),
        n_edges=n_edges,
        neighbor_quantiles=np.quantile(degree[eligible], [0, .5, .9, .99, 1]).tolist(),
    )

    #Old, claude:
    # cfg=configuration(run);out=Path(run)/'prepared';xy=array(run,'xy_um');sec=array(run,'sections');nonempty=array(run,'nonempty');n=len(xy)
    # degree=np.zeros(n,np.int64);half=cfg['patch_width_um']/2
    # for s in np.unique(sec):
    #     ids=np.flatnonzero(sec==s);tree=cKDTree(xy[ids])
    #     for a,b in chunks(len(ids),4096):degree[ids[a:b]]=tree.query_ball_point(xy[ids[a:b]],half,p=np.inf,return_length=True,workers=cfg['threads'])-1
    # ptr=np.r_[0,np.cumsum(degree)];require((degree>=0).all(),'Invalid degree')
    # needed=int(ptr[-1])*4+n*384*8+int(cfg.get('storage_reserve_gib',20))*1024**3
    # require(shutil.disk_usage(out).free>needed,f'Graph and two embedding arrays need about {needed/1024**3:.1f} GiB additional free space')
    # atomic_npy(out/'neighbor_indptr.npy',ptr);edges=mmap_create(out/'neighbor_indices.npy',(int(ptr[-1]),),np.int32)
    # for s in np.unique(sec):
    #     ids=np.flatnonzero(sec==s);tree=cKDTree(xy[ids]);order=np.argsort(xy[ids,0],kind='stable');rank=np.empty(len(ids),np.int64);rank[order]=np.arange(len(ids))
    #     for a,b in chunks(len(ids),2048):
    #         near=tree.query_ball_point(xy[ids[a:b]],half,p=np.inf,workers=cfg['threads'])
    #         for i,local in zip(ids[a:b],near):
    #             local=np.asarray(local,np.int64);local=local[ids[local]!=i];local=local[np.argsort(rank[local],kind='stable')];edges[ptr[i]:ptr[i+1]]=ids[local]
    #     log(f'Cached CT2 neighborhoods for section {s+1}/{len(np.unique(sec))}')
    # edges.flush();eligible=degree>0;atomic_npy(out/'eligible.npy',eligible)
    # rng=np.random.default_rng(cfg['seed']);ids=np.flatnonzero(eligible)
    # require(len(ids)>1,'Fewer than two eligible centers after the cell-ID join/neighborhood construction')
    # val=np.sort(rng.choice(ids,max(1,round(len(ids)*cfg['validation_fraction'])),replace=False)).astype(np.int64)
    # mask=eligible.copy();mask[val]=False;train=np.flatnonzero(mask)
    # require(len(train)>0 and len(val)>0,'Empty split')
    # atomic_npy(out/'train_ids.npy',train);atomic_npy(out/'validation_ids.npy',val)
    # return dict(n_eligible=int(eligible.sum()),n_train=len(train),n_validation=len(val),n_edges=int(ptr[-1]),neighbor_quantiles=np.quantile(degree[eligible],[0,.5,.9,.99,1]).tolist())



def prepare(run):
    """
    Full Allen 1 preprocessing: load annotations and raw counts, join them by
    cell ID, export counts / genes / cell metadata / label codes, build the
    neighborhood graph and train/validation split, and write a summary.
    """
    
    run = Path(run)
    cfg = configuration(run)
    raw = Path(cfg['root']) / 'data/raw'
    out = run / 'prepared'
    out.mkdir(exist_ok=True)
    #software_test runs use small fake data, so skip checks against the real dataset sizes
    production = not cfg.get('software_test', False)

    log('Reading pinned cell-type and CCF annotations')
    meta = load_metadata(raw / 'cells_ccf.csv')
    n_annotated = len(meta)

    with h5py.File(raw / 'expression_raw.h5ad', 'r') as f:
        #check count matrix format
        x = f['X']
        if x.attrs.get('encoding-type') != 'csr_matrix':
            raise ValueError('Expected released CSR expression matrix')
        shape = tuple(int(v) for v in x.attrs['shape'])

        #get count matrix cell IDs
        obs = f['obs']
        raw_ids = obs[obs.attrs['_index']].asstr()[:]

        genes, symbols, chosen = select_genes(f)
        n_genes_kept = int(chosen.sum())

        rows, meta, missing = align_metadata(raw_ids, meta)
        if len(rows) < 2:
            raise ValueError('The count file and annotated-cell table have fewer than two shared cell IDs')

        #check sizes match the pinned release
        if production:
            if shape != (EXPECTED['raw_cells'], EXPECTED['raw_channels']):
                raise ValueError('Unexpected raw matrix inventory')
            if n_annotated != EXPECTED['annotated_cells']:
                raise ValueError('Unexpected annotation row count')
            if len(rows) != EXPECTED['cells'] or n_genes_kept != EXPECTED['genes']:
                raise ValueError(f'Pinned release cohort/gene inventory mismatch: {len(rows)} cells, {n_genes_kept} genes')
            if meta.brain_section_label.nunique() != EXPECTED['sections']:
                raise ValueError('Unexpected section count')
            if meta.subclass.nunique() != EXPECTED['subclasses']:
                raise ValueError('Unexpected subclass count')

        #save cell ID join results
        missing.to_csv(out / 'annotation_ids_without_counts.tsv', sep='\t', index=False)
        atomic_npy(out / 'raw_row_indices.npy', rows)

        #save gene table and kept genes
        gene_table = pd.DataFrame(dict(
            gene_identifier=genes,
            gene_symbol=symbols,
            selected=chosen,
            reason=np.where(chosen, 'Measured biological gene', 'Blank control probe'),
        ))
        gene_table.to_csv(out / 'gene_selection.tsv', sep='\t', index=False)
        json_write(out / 'genes.json', genes[chosen].tolist())
        json_write(out / 'gene_symbols.json', symbols[chosen].tolist())

        #dense int32 counts on disk (kept cells x kept genes), plus per-cell totals
        n = len(rows)
        target = mmap_create(out / 'counts.npy', (n, n_genes_kept), np.int32)
        total = np.zeros(n, np.int64)
        ngenes = np.zeros(n, np.int32)

        #read the raw matrix in consecutive row blocks (fast, never loads the whole matrix)
        for a, b in chunks(shape[0], 8192):
            #kept rows that fall in this block (rows is sorted)
            lo, hi = np.searchsorted(rows, [a, b])
            local = rows[lo:hi] - a
            if not len(local):
                continue
            block = csr_block(x, a, b)[local][:, chosen].toarray()

            #check counts are finite, nonnegative integers that fit in int32
            is_valid = np.isfinite(block).all() and (block >= 0).all() and np.equal(block, np.floor(block)).all()
            if not is_valid:
                raise ValueError('Raw molecule counts must be finite, nonnegative integers; no rounding is applied')
            if block.max(initial=0) > np.iinfo(np.int32).max:
                raise ValueError('Count overflow')

            target[lo:hi] = block.astype(np.int32)
            total[lo:hi] = block.sum(1, dtype=np.int64)
            ngenes[lo:hi] = (block > 0).sum(1)

            #log progress every ~262k raw cells
            if a // 262144 != b // 262144:
                log(f'Counts: scanned {b:,}/{shape[0]:,} raw cells')
        target.flush()

    #rename columns to the names used downstream
    meta = meta.rename(columns={
        'cell_label': 'cell_id',
        'brain_section_label': 'section',
        'class': 'cell_class',
        'subclass': 'cell_subclass',
        'cluster': 'cell_cluster',
    })

    #coordinates: mm -> um
    xy = meta[['x_reconstructed', 'y_reconstructed']].to_numpy(np.float64) * 1000.
    if not np.isfinite(xy).all():
        raise ValueError('Nonfinite physical coordinates')

    #add per-cell columns
    meta['x_um'] = xy[:, 0]
    meta['y_um'] = xy[:, 1]
    meta['total_counts'] = total
    meta['n_genes'] = ngenes
    meta['reference_valid'] = True

    vocab = references(meta, out)

    #save section and cell type labels as integer codes
    #column -> output file name
    label_files = {
        'section': 'sections',
        'cell_subclass': 'types',
        'cell_cluster': 'cell_cluster',
        'cell_class': 'cell_class',
    }
    for column, file_name in label_files.items():
        categories = sorted(meta[column].unique().tolist())
        if '' in categories:
            raise ValueError('Empty cell type annotation')
        vocab[column] = categories
        codes = pd.Categorical(meta[column], categories=categories).codes.astype(np.int32)
        atomic_npy(out / (file_name + '.npy'), codes)

    if len(vocab['cell_subclass']) > cfg['model']['cell_type_cardinality']:
        raise ValueError('Cell types exceed model capacity')

    json_write(out / 'vocabulary.json', vocab)
    atomic_npy(out / 'xy_um.npy', xy)
    atomic_npy(out / 'nonempty.npy', total > 0)
    meta.to_parquet(out / 'cells.parquet', index=False)

    graph = graph_and_split(run)

    #check every CCF label appears among eligible centers
    eligible = array(run, 'eligible')
    for level, k in LEVELS.items():
        observed = len(np.unique(array(run, 'reference_' + level)[eligible]))
        if production and observed != k:
            raise ValueError(f'Expected {k} populated {level} labels on eligible centers, found {observed}')

    #per-section cell and count totals
    inventory = meta.groupby('section', observed=True).agg(
        cells=('cell_id', 'size'),
        total_counts=('total_counts', 'sum'),
    )
    inventory.to_csv(out / 'section_inventory.tsv', sep='\t')

    summary = dict(
        paper_reported_cells=3737550,
        released_annotation_cells=3739961,
        paper_count_difference_note='Released annotation/matrix intersection has 2,411 more rows than the reported paper total.'
                                    ' Exact original analysis cell list is not established by these release files;'
                                    ' all verified rows retained and eligible count reported separately.',
        dataset='Allen 1 / MERFISH-C57BL6J-638850',
        release='20231215',
        raw_matrix_shape=shape,
        n_annotation_rows=n_annotated,
        n_annotation_ids_without_counts=len(missing),
        n_cells=n,
        n_sections=len(vocab['section']),
        n_model_genes=n_genes_kept,
        n_model_cell_types=len(vocab['cell_subclass']),
        n_mice=1,
        n_zero_count_cells=int((total == 0).sum()),
        n_excluded_without_neighbors=int((~eligible).sum()),
        zero_count_policy='Retained as context and centers if they have neighbors',
        **graph,
        coordinate_conversion='x_reconstructed/y_reconstructed in mm multiplied by 1000;'
                              ' CCF-warped coordinates are not used for model neighborhoods',
        gene_order='Original raw H5 order, excluding 50 Blank controls; Ensembl IDs and symbols exported',
        cell_order='Intersection of released count IDs and 20231215 annotated/registered IDs, in count-matrix order',
        reference_counts={level: len(vocab['reference_' + level]) for level in LEVELS},
        validation='Random 80/20 split of eligible centers across the dataset;'
                   ' spatial context overlaps; not held-out-section/animal evaluation',
        software_test=not production,
    )
    json_write(out / 'summary.json', summary)
    log('ALLEN 1 PREPARATION: PASSED')
    log(json.dumps(summary))

    #Old, claude:
    # run=Path(run);cfg=configuration(run);raw=Path(cfg['root'])/'data/raw';out=run/'prepared';out.mkdir(exist_ok=True)
    # production=not cfg.get('software_test',False)
    # log('Reading pinned cell-type and CCF annotations')
    # meta=load_metadata(raw/'cells_ccf.csv');n_annotated=len(meta)
    # with h5py.File(raw/'expression_raw.h5ad','r') as f:
    #     x=f['X'];require(x.attrs.get('encoding-type')=='csr_matrix','Expected released CSR expression matrix')
    #     shape=tuple(int(v) for v in x.attrs['shape']);raw_ids=f['obs'][f['obs'].attrs['_index']].asstr()[:]
    #     genes,symbols,chosen=select_genes(f)
    #     rows,meta,missing=align_metadata(raw_ids,meta)
    #     require(len(rows)>1,'The count file and annotated-cell table have fewer than two shared cell IDs')
    #     if production:
    #         require(shape==(EXPECTED['raw_cells'],EXPECTED['raw_channels']),'Unexpected raw matrix inventory')
    #         require(n_annotated==EXPECTED['annotated_cells'],'Unexpected annotation row count')
    #         require(len(rows)==EXPECTED['cells'] and int(chosen.sum())==EXPECTED['genes'],
    #                 f'Pinned release cohort/gene inventory mismatch: {len(rows)} cells, {chosen.sum()} genes')
    #         require(meta.brain_section_label.nunique()==EXPECTED['sections'],'Unexpected section count')
    #         require(meta.subclass.nunique()==EXPECTED['subclasses'],'Unexpected subclass count')
    #     missing.to_csv(out/'annotation_ids_without_counts.tsv',sep='\t',index=False)
    #     atomic_npy(out/'raw_row_indices.npy',rows)
    #     pd.DataFrame(dict(gene_identifier=genes,gene_symbol=symbols,selected=chosen,
    #                      reason=np.where(chosen,'Measured biological gene','Blank control probe'))).to_csv(out/'gene_selection.tsv',sep='\t',index=False)
    #     json_write(out/'genes.json',genes[chosen].tolist());json_write(out/'gene_symbols.json',symbols[chosen].tolist())
    #     n=len(rows);target=mmap_create(out/'counts.npy',(n,int(chosen.sum())),np.int32);total=np.zeros(n,np.int64);ngenes=np.zeros(n,np.int32)
    #     # Consecutive CSR reads are fast and do not load the full sparse matrix.
    #     for a,b in chunks(shape[0],8192):
    #         lo,hi=np.searchsorted(rows,[a,b]);local=rows[lo:hi]-a
    #         if not len(local):continue
    #         block=csr_block(x,a,b)[local][:,chosen].toarray()
    #         require(np.isfinite(block).all() and (block>=0).all() and np.equal(block,np.floor(block)).all(),
    #                 'Raw molecule counts must be finite, nonnegative integers; no rounding is applied')
    #         require(block.max(initial=0)<=np.iinfo(np.int32).max,'Count overflow')
    #         target[lo:hi]=block.astype(np.int32);total[lo:hi]=block.sum(1,dtype=np.int64);ngenes[lo:hi]=(block>0).sum(1)
    #         if a//262144 != b//262144:log(f'Counts: scanned {b:,}/{shape[0]:,} raw cells')
    #     target.flush()
    # meta=meta.rename(columns={'cell_label':'cell_id','brain_section_label':'section','class':'cell_class',
    #                          'subclass':'cell_subclass','cluster':'cell_cluster'})
    # xy=meta[['x_reconstructed','y_reconstructed']].to_numpy(np.float64)*1000.
    # require(np.isfinite(xy).all(),'Nonfinite physical coordinates')
    # meta['x_um']=xy[:,0];meta['y_um']=xy[:,1];meta['total_counts']=total;meta['n_genes']=ngenes
    # meta['reference_valid']=True
    # vocab=references(meta,out)
    # for column,name in [('section','section'),('cell_subclass','cell_subclass'),('cell_cluster','cell_cluster'),('cell_class','cell_class')]:
    #     cats=sorted(meta[column].unique().tolist());require('' not in cats,'Empty cell type annotation')
    #     vocab[name]=cats
    #     code=pd.Categorical(meta[column],categories=cats).codes.astype(np.int32)
    #     atomic_npy(out/({'section':'sections','cell_subclass':'types'}.get(name,name)+'.npy'),code)
    # require(len(vocab['cell_subclass'])<=cfg['model']['cell_type_cardinality'],'Cell types exceed model capacity')
    # json_write(out/'vocabulary.json',vocab);atomic_npy(out/'xy_um.npy',xy);atomic_npy(out/'nonempty.npy',total>0)
    # meta.to_parquet(out/'cells.parquet',index=False)
    # graph=graph_and_split(run)
    # eligible=array(run,'eligible')
    # for level,k in LEVELS.items():
    #     observed=len(np.unique(array(run,'reference_'+level)[eligible]))
    #     if production:require(observed==k,f'Expected {k} populated {level} labels on eligible centers, found {observed}')
    # inventory=meta.groupby('section',observed=True).agg(cells=('cell_id','size'),total_counts=('total_counts','sum'))
    # inventory.to_csv(out/'section_inventory.tsv',sep='\t')
    # summary=dict(paper_reported_cells=3737550, released_annotation_cells=3739961, paper_count_difference_note='Released annotation/matrix intersection has 2,411 more rows than the reported paper total. Exact original analysis cell list is not established by these release files; all verified rows retained and eligible count reported separately.',dataset='Allen 1 / MERFISH-C57BL6J-638850',release='20231215',raw_matrix_shape=shape,
    #     n_annotation_rows=n_annotated,n_annotation_ids_without_counts=len(missing),n_cells=n,n_sections=len(vocab['section']),
    #     n_model_genes=int(chosen.sum()),n_model_cell_types=len(vocab['cell_subclass']),n_mice=1,
    #     n_zero_count_cells=int((total==0).sum()),n_excluded_without_neighbors=int((~eligible).sum()),zero_count_policy='Retained as context and centers if they have neighbors',
    #     **graph,coordinate_conversion='x_reconstructed/y_reconstructed in mm multiplied by 1000; CCF-warped coordinates are not used for model neighborhoods',
    #     gene_order='Original raw H5 order, excluding 50 Blank controls; Ensembl IDs and symbols exported',
    #     cell_order='Intersection of released count IDs and 20231215 annotated/registered IDs, in count-matrix order',
    #     reference_counts={level:len(vocab['reference_'+level]) for level in LEVELS},
    #     validation='Random 80/20 split of eligible centers across the dataset; spatial context overlaps; not held-out-section/animal evaluation',
    #     software_test=not production)
    # json_write(out/'summary.json',summary);log('ALLEN 1 PREPARATION: PASSED');log(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run)

    #Old, claude:
    # p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run)
