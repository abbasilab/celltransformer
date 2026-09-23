"""Pinned BARseq pilot counts, published cell types and registered anatomy.

The authors' IDs, not row positions, join independent released files. Cell-type
labels are expression-derived inputs; anatomical labels are evaluation-only.
"""
import argparse, gzip, shutil, subprocess
import pandas as pd
from scipy import sparse
from scipy.io import loadmat
from scipy.spatial import cKDTree
from common import *


def fetch(spec, dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    marker=dest.with_suffix(dest.suffix+'.verified.json')
    if marker.exists() and dest.exists():
        old=read_json(marker)
        if old['spec']==spec and old['size']==dest.stat().st_size and old['mtime_ns']==dest.stat().st_mtime_ns:return
    if not dest.exists():
        partial=dest.with_suffix(dest.suffix+'.partial')
        # Retry/restart is explicit; an invalid complete file is never overwritten.
        if not partial.exists() or partial.stat().st_size<spec['size']:
            subprocess.run(['curl','--fail','--location','--retry','8','--retry-delay','5',
                '--connect-timeout','60','--continue-at','-','--output',str(partial),spec['url']],check=True)
        require(partial.stat().st_size==spec['size'],'Download size mismatch: '+str(partial))
        require(digest(partial)==spec['sha256'],'Download SHA256 mismatch: '+str(partial))
        partial.replace(dest)
    require(dest.stat().st_size==spec['size'] and digest(dest)==spec['sha256'],'Pinned file changed: '+str(dest))
    json_write(marker,dict(spec=spec,size=dest.stat().st_size,mtime_ns=dest.stat().st_mtime_ns))


def ontology():
    nodes={}
    def walk(node,anc):
        nodes[node['id']]={**{k:v for k,v in node.items() if k!='children'},'path':anc+[node['id']]}
        for child in node.get('children',[]):walk(child,anc+[node['id']])
    for root in read_json(HERE/'reference/allen_structure_graph.json')['msg']:walk(root,[])
    return nodes


def references(meta,out):
    nodes=ontology();ids=meta.gene_area.to_numpy(np.int64);valid=meta.reference_valid.to_numpy(bool)
    require(set(ids[valid])<=set(nodes),'Unknown CCF IDs in reference')
    vocab={};refinfo={}
    for name,depth in [('depth5',5),('depth7',7),('released',None)]:
        mapping={i:(int(i) if depth is None else nodes[int(i)]['path'][min(depth,len(nodes[int(i)]['path']))-1]) for i in np.unique(ids[valid])}
        codes=sorted(set(mapping.values()));index={v:k for k,v in enumerate(codes)}
        values=np.full(len(meta),-1,np.int32)
        # Released parent IDs are residual, disjoint assignments, not unions with
        # their descendants. Explicit ancestor coarsening forms a partition.
        for i in mapping:values[valid&(ids==i)]=index[mapping[i]]
        atomic_npy(out/f'reference_{name}.npy',values)
        names=[nodes[i]['acronym'] for i in codes];vocab['reference_'+name]=names
        refinfo[name]=dict(k=len(codes),ids=codes,names=names,depth=depth,colors=['#'+nodes[i]['color_hex_triplet'] for i in codes],
            label='Released CCF region assignments' if depth is None else f'CCF hierarchy depth {depth} (root=1)',
            definition='Per-cell region names in authors\' neurons_registered.csv, mapped by exact name to Allen ontology; parent IDs represent residual assignments; includes non-grey-matter structures' if depth is None else 'Ancestor at the stated tree depth; root=1; stop at shallower labels. Not the MERFISH paper\'s division/structure/substructure cuts')
    json_write(out/'references.json',refinfo);return vocab


def read_atlas(path):
    header,payload=Path(path).read_bytes().split(b'\n\n',1)
    require(all(v in header for v in [b'type: unsigned int',b'sizes: 528 320 456',b'encoding: gzip',b'endian: little']),'Unexpected CCF volume format')
    return np.frombuffer(gzip.decompress(payload),dtype='<u4').reshape((528,320,456),order='F')


def lookup_atlas(ccf,volume):
    """Author coordinates are already zero-based 25-um Allen voxel coordinates."""
    finite_rows=np.isfinite(ccf).all(1);v=np.zeros(ccf.shape,np.int64);v[finite_rows]=np.floor(ccf[finite_rows]+.5).astype(np.int64)
    valid=finite_rows&(v>=0).all(1)&(v<np.array(volume.shape)).all(1)
    ids=np.zeros(len(ccf),np.int64);ids[valid]=volume[tuple(v[valid].T)]
    return ids


def select_cells(m,labels):
    ids=np.asarray(m['id'],np.int64);require(len(np.unique(ids))==len(ids),'Duplicate matrix cell IDs')
    parts=labels['sample'].str.extract(r'^(\d+)_(\d+)$');require(parts.notna().all().all(),'Malformed label sample IDs')
    label_ids=parts[1].astype(np.int64);require(not label_ids.duplicated().any(),'Duplicate label cell IDs')
    index=pd.Index(label_ids).get_indexer(ids);keep=index>=0
    require(np.array_equal(keep,np.asarray(m['clustid'])!='qc-filtered'),'Published QC label join disagrees with released matrix QC assignments')
    selected=labels.iloc[index[keep]].reset_index(drop=True)
    require(np.array_equal(parts[0].iloc[index[keep]].astype(int),m['slice'][keep]),'Cell-ID join crosses sections')
    missing=labels.loc[~label_ids.isin(ids)].copy()
    return keep,selected,missing


def select_genes(m):
    genes=np.asarray(m['genes'])[:,0].tolist();chosen=read_json(HERE/'reference/sequencing_genes.json')
    require(len(set(genes))==len(genes),'Duplicate matrix gene names')
    columns=pd.Index(genes).get_indexer(chosen);require((columns>=0).all(),'A pinned sequencing gene is absent')
    require(not any(g.startswith('unused') or g in ['Slc17a7','Gad1','Slc30a3'] for g in chosen),'Control/hybridization channel selected')
    return genes,chosen,columns


def annotate_types(labels):
    meta=labels.rename(columns={'class':'cell_class','subclass':'cell_subclass','cluster':'cell_cluster'}).copy()
    for field in ['cell_subclass','cell_cluster']:
        absent=meta[field].isin(['NA','']);meta[field+'_fallback_to_class']=absent
        meta.loc[absent,field]=meta.loc[absent,'cell_class']
    require(not meta.cell_class.isin(['NA','']).any(),'Missing class cannot supply fallback type')
    meta['qc_flag']=meta.cell_subclass.eq('Low Quality')
    return meta


def graph_and_split(run):
    cfg=configuration(run);out=Path(run)/'prepared';xy=array(run,'xy_um');sec=array(run,'sections');nonempty=array(run,'nonempty');n=len(xy)
    degree=np.zeros(n,np.int64);half=cfg['patch_width_um']/2
    for s in np.unique(sec):
        ids=np.flatnonzero((sec==s)&nonempty);tree=cKDTree(xy[ids])
        for a,b in chunks(len(ids),4096):degree[ids[a:b]]=tree.query_ball_point(xy[ids[a:b]],half,p=np.inf,return_length=True,workers=cfg['threads'])-1
    ptr=np.r_[0,np.cumsum(degree)];require((degree>=0).all(),'Invalid degree')
    needed=int(ptr[-1])*4+n*384*8+int(cfg.get('storage_reserve_gib',20))*1024**3
    require(shutil.disk_usage(out).free>needed,f'Graph and two embedding arrays need about {needed/1024**3:.1f} GiB additional free space')
    atomic_npy(out/'neighbor_indptr.npy',ptr);edges=mmap_create(out/'neighbor_indices.npy',(int(ptr[-1]),),np.int32)
    for s in np.unique(sec):
        ids=np.flatnonzero((sec==s)&nonempty);tree=cKDTree(xy[ids]);order=np.argsort(xy[ids,0],kind='stable');rank=np.empty(len(ids),np.int64);rank[order]=np.arange(len(ids))
        for a,b in chunks(len(ids),2048):
            near=tree.query_ball_point(xy[ids[a:b]],half,p=np.inf,workers=cfg['threads'])
            for i,local in zip(ids[a:b],near):
                local=np.asarray(local,np.int64);local=local[ids[local]!=i];local=local[np.argsort(rank[local],kind='stable')];edges[ptr[i]:ptr[i+1]]=ids[local]
        log(f'Cached CT2 neighborhoods for section {s+1}/{len(np.unique(sec))}')
    edges.flush();eligible=nonempty&(degree>0);atomic_npy(out/'eligible.npy',eligible)
    rng=np.random.default_rng(cfg['seed']);val=[]
    for s in np.unique(sec):
        ids=np.flatnonzero((sec==s)&eligible)
        if len(ids)>1:val.extend(rng.choice(ids,min(len(ids)-1,max(1,round(len(ids)*cfg['validation_fraction']))),replace=False))
    val=np.sort(val).astype(np.int64);mask=eligible.copy();mask[val]=False;train=np.flatnonzero(mask)
    require(len(train)>0 and len(val)>0,'Empty split')
    atomic_npy(out/'train_ids.npy',train);atomic_npy(out/'validation_ids.npy',val)
    return dict(n_eligible=int(eligible.sum()),n_train=len(train),n_validation=len(val),n_edges=int(ptr[-1]),neighbor_quantiles=np.quantile(degree[eligible],[0,.5,.9,.99,1]).tolist())


def main(run):
    run=Path(run);cfg=configuration(run);root=Path(cfg['root']);raw=root/'data/raw';out=run/'prepared';out.mkdir(exist_ok=True)
    for spec in read_json(HERE/'DOWNLOAD_MANIFEST.json')['files']:
        log('Download/checksum: '+spec['name']);fetch(spec,raw/spec['name'])
    log('Loading released sparse counts and joining published cell labels')
    m=loadmat(raw/'wholecortexBARseq_filt_neurons_fixedbent_CCF.mat',simplify_cells=True)['filt_neurons']
    labels=pd.read_csv(raw/'labels_20211208_withgaba_nonexc.csv',keep_default_na=False)
    keep,selected,missing=select_cells(m,labels)
    require(len(m['id'])==2167762 and len(labels)==1259256 and int(keep.sum())==1161387,'Pinned cohort inventory changed')
    missing.to_csv(out/'published_label_ids_without_released_counts.tsv',sep='\t',index=False)
    meta=annotate_types(selected);meta['cell_id']=m['id'][keep].astype(np.int64);meta['section_number']=m['slice'][keep].astype(int)
    meta['section']=meta.section_number.map(lambda x:f'Slice{x:02d}')
    xy=np.asarray(m['fixedpos'][keep],np.float64);require(np.isfinite(xy).all(),'Nonfinite physical cell coordinates')
    meta['x_um']=xy[:,0];meta['y_um']=xy[:,1]
    genes,chosen,columns=select_genes(m);x=sparse.csr_matrix(m['expmat'][keep][:,columns]);x.eliminate_zeros()
    require(np.isfinite(x.data).all() and (x.data>=0).all() and np.equal(x.data,np.floor(x.data)).all(),'Targets must be integer nonnegative molecule counts')
    require(x.data.max()<=np.iinfo(np.int32).max,'Count integer overflow')
    n=len(meta);target=mmap_create(out/'counts.npy',(n,len(chosen)),np.int32)
    for a,b in chunks(n,4096):target[a:b]=x[a:b].toarray().astype(np.int32)
    target.flush();meta['total_counts']=np.asarray(x.sum(1)).ravel().astype(np.int64);meta['n_genes']=np.diff(x.indptr)
    nonempty=meta.total_counts.to_numpy()>0
    stats=pd.DataFrame(dict(gene=genes,selected=[v in chosen for v in genes]));stats['reason']=np.where(stats.selected,'Released named sequencing target',np.where(stats.gene.str.startswith('unused'),'Unused barcode control','Separate hybridization marker channel'))
    stats.to_csv(out/'gene_selection.tsv',sep='\t',index=False);json_write(out/'genes.json',chosen)
    # Only required columns read; no reliance on anatomical file row order.
    reg=pd.read_csv(raw/'neurons_registered.csv',usecols=['id','slice','pos_x','pos_y','index','name','x_CCF','y_CCF','z_CCF'],keep_default_na=False)
    require(not reg.id.duplicated().any(),'Duplicate registration IDs')
    ri=pd.Index(reg.id).get_indexer(meta.cell_id);require((ri>=0).all(),'Retained cells lack registration records')
    reg=reg.iloc[ri].reset_index(drop=True)
    require(np.array_equal(reg['slice'],meta.section_number),'Registration IDs cross sections')
    require(np.allclose(reg[['pos_x','pos_y']],xy,rtol=0,atol=1e-5),'Registration and matrix physical positions disagree')
    ccf=m['CCF'][keep];require(np.allclose(reg[['x_CCF','y_CCF','z_CCF']],ccf,rtol=0,atol=1e-8,equal_nan=True),'Registration and matrix CCF coordinates disagree')
    meta['released_region_name']=reg['name'];meta['released_region_palette_index']=reg['index']
    nodes=ontology();byname={v['name']:i for i,v in nodes.items()}
    names=set(reg['name']);missing_names=sorted(names-set(byname)-{'','background','Background','not found','root','empty'})
    require(not missing_names,'Unmapped released region names: '+str(missing_names))
    meta['gene_area']=reg['name'].map(byname).fillna(0).astype(np.int64)
    meta['reference_valid']=(meta.gene_area!=0)&(meta.gene_area!=997)
    atlas=read_atlas(raw/'annotation_25.nrrd');voxel=lookup_atlas(ccf,atlas)
    meta['ccf_voxel_id_audit']=voxel
    for i,name in enumerate(['ccf_ap_25um','ccf_dv_25um','ccf_ml_25um']):meta[name]=ccf[:,i]
    # The independent voxel lookup is an audit, not a replacement for the released
    # nonlinear 2D annotation. Boundary mismatches are preserved, never optimised.
    valid=meta.reference_valid.to_numpy();agreement=float(np.mean(voxel[valid]==meta.gene_area.to_numpy()[valid]))
    vocab=references(meta,out)
    for column,name in [('section','section'),('cell_subclass','cell_subclass'),('cell_cluster','cell_cluster'),('cell_class','cell_class')]:
        cats=sorted(meta[column].unique());vocab[name]=cats
        code=pd.Categorical(meta[column],categories=cats).codes.astype(np.int32)
        atomic_npy(out/({'section':'sections','cell_subclass':'types'}.get(name,name)+'.npy'),code)
    atomic_npy(out/'xy_um.npy',xy);atomic_npy(out/'nonempty.npy',nonempty)
    meta.to_parquet(out/'cells.parquet',index=False);json_write(out/'vocabulary.json',vocab)
    cfg['model'].update(expression_dim=len(chosen),cell_type_cardinality=len(vocab['cell_subclass']));json_write(run/'config.json',cfg)
    sections=meta.groupby('section',sort=True).agg(n_cells=('cell_id','size'),n_reference_valid=('reference_valid','sum'),n_qc_flagged=('qc_flag','sum')).reset_index();sections.to_csv(out/'section_inventory.tsv',sep='\t',index=False)
    summary=graph_and_split(run)
    summary.update(n_cells=n,n_nonempty=int(nonempty.sum()),n_isolated_nonempty=int(nonempty.sum())-summary['n_eligible'],n_sections=len(vocab['section']),n_mice=1,n_measured_columns=len(genes),n_model_genes=len(chosen),n_cell_subclasses=len(vocab['cell_subclass']),n_cell_clusters=len(vocab['cell_cluster']),n_qc_flagged=int(meta.qc_flag.sum()),n_reference_valid=int(valid.sum()),
        release='BARseq pilot / Mendeley 8bhhk7c5n9 version 1',paper_cells=2167762,paper_high_quality_cells=1259256,n_published_labels_without_released_counts=len(missing),
        release_count_discrepancy='1,161,387 published QC-labelled IDs occur in the overlap-filtered released matrix. The 97,869 label IDs without counts are inventoried, not silently assigned or imputed. Retained IDs exactly equal the matrix non-qc-filtered set.',
        gene_panel='106 named sequencing columns in released matrix; exclude 5 unused barcodes and Slc17a7/Gad1/Slc30a3 hybridization channels. Paper says 104 genes; author GitHub and supplementary panel lists differ. No attempt to force that number.',
        input='log2(integer raw molecule counts + 1); no library normalization, rounding or HVG selection',
        type_annotations='Published subclass; missing values fall back to released expression-derived SubCTX class. Same rule for cluster-level composition. Types with regional names are author expression clusters, not CCF region IDs.',
        split='95/5 eligible centers within each section. Validation centers can appear in training neighborhoods; exploratory context-shared validation, not independent-mouse generalization.',
        reference_use='Authors\' registered region names mapped to Allen IDs; used only after training. Voxel lookup is a separate audit.',
        ccf_voxel_exact_agreement=agreement,coordinate_unit='Micrometers; released fixedpos (authors\' corrected 2D coordinates), not atlas-warped training coordinates',patch_width_um=cfg['patch_width_um'],section_summaries=sections.to_dict('records'))
    json_write(out/'summary.json',summary);log('BARSEQ PREPARATION AND CT2 NEIGHBORHOODS: PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
