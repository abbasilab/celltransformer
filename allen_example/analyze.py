"""Section-safe Gaussian smoothing and Figure 2c–i analogue metrics."""
import argparse
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from common import *

METHODS={'celltransformer':'CellTransformer (-smoothing)','celltransformer_smoothed':'CellTransformer (+smoothing)'}


def prepare_references(run):
    out=Path(run)/'analysis';out.mkdir(exist_ok=True);eligible=array(run,'eligible');refs=read_json(Path(run)/'prepared/references.json');valid=eligible.copy()
    for name in refs:valid &= array(run,'reference_'+name)>=0
    ids=np.flatnonzero(valid);require(len(ids)>0,'No common cells with anatomical reference');atomic_npy(out/'evaluation_ids.npy',ids)
    for name,info in refs.items():
        y=array(run,'reference_'+name)[ids];present=np.unique(y);lookup=np.full(info['k'],-1,np.int32);lookup[present]=np.arange(len(present));atomic_npy(out/f'reference_{name}.npy',lookup[y])
        for field in ['ids','names','colors']:info[field]=[info[field][i] for i in present]
        info['k']=len(present)
    json_write(out/'references.json',refs)


def smooth_section(xy,embedding,sigma,k,device='cpu',chunk=128):
    """Same native Gaussian kernel; local indices always index local embeddings."""
    import torch
    from celltransformer.smooth.main import _gaussian_weights
    tree=cKDTree(xy);k=min(k,len(xy));source=torch.as_tensor(np.asarray(embedding),device=device);output=np.empty_like(embedding)
    for a,b in chunks(len(xy),chunk):
        d,near=tree.query(xy[a:b],k=k,workers=1)
        d=np.asarray(d).reshape(b-a,k);near=np.asarray(near).reshape(b-a,k)
        with torch.no_grad():
            w=_gaussian_weights(torch.tensor(d,dtype=torch.float32,device=device),sigma)
            values=(source[torch.tensor(near,device=device)]*w[...,None]).sum(1)
            output[a:b]=values.cpu().numpy()
    return output


def smooth(run):
    cfg=configuration(run);out=Path(run)/'analysis';out.mkdir(exist_ok=True);prepare_references(run)
    eligible=array(run,'eligible');xy=array(run,'xy_um');sec=array(run,'sections');embedding=np.load(Path(run)/'celltransformer/embeddings.npy',mmap_mode='r')
    dest=out/'smoothed.npy';progress=out/'smoothing_progress.json';completed=[]
    if progress.exists():
        completed=read_json(progress)['completed_sections'];target=np.lib.format.open_memmap(dest,mode='r+')
    else:
        target=mmap_create(dest,embedding.shape,np.float32)
        for a,b in chunks(len(target)):target[a:b]=np.nan
    for s in np.unique(sec):
        if int(s) in completed:continue
        ids=np.flatnonzero((sec==s)&eligible)
        if len(ids):target[ids]=smooth_section(xy[ids],embedding[ids],cfg['smoothing']['sigma_um'],cfg['smoothing']['k'],cfg['device'])
        target.flush();completed.append(int(s));json_write(progress,dict(completed_sections=completed))
        log(f'Gaussian smoothing: section {int(s)+1}/{len(np.unique(sec))}, {len(ids):,} centers')
    json_write(out/'smoothing.json',dict(**cfg['smoothing'],neighbors='Exact Euclidean same-section kNN (scipy cKDTree)',kernel='Unchanged native CT2 _gaussian_weights',
        adapter='Section-local indexing and chunked gathers correct the supplied CLI indexing error and limit memory'))


def homogeneity(labels,neighbors,k):
    sums=np.zeros(k);sizes=np.zeros(k,np.int64)
    for a,b in chunks(len(labels),8192):
        near=neighbors[a:b];valid=near>=0;den=valid.sum(1);score=np.divide(((labels[np.maximum(near,0)]==labels[a:b,None])&valid).sum(1),den,out=np.full(b-a,np.nan),where=den>0)
        ok=np.isfinite(score);sums+=np.bincount(labels[a:b][ok],weights=score[ok],minlength=k);sizes+=np.bincount(labels[a:b][ok],minlength=k)
    per=np.divide(sums,sizes,out=np.full(k,np.nan),where=sizes>0)
    return float(sums.sum()/sizes.sum()),per,sizes


def composition(labels,types,k,nt):
    counts=np.bincount(labels.astype(np.int64)*nt+types,minlength=k*nt).reshape(k,nt);sizes=counts.sum(1)
    return np.divide(counts,sizes[:,None],out=np.zeros_like(counts,dtype=float),where=sizes[:,None]>0),sizes


def correlations(a,b):
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True);den=np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:]
    return np.clip(np.divide(a@b.T,den,out=np.full(den.shape,np.nan),where=den>0),-1,1)


def match(corr):
    rows=np.flatnonzero(np.isfinite(corr).any(1));cols=np.flatnonzero(np.isfinite(corr).any(0));pairs=[]
    if len(rows) and len(cols):
        a,b=linear_sum_assignment(np.where(np.isfinite(corr[np.ix_(rows,cols)]),-corr[np.ix_(rows,cols)],1e6))
        pairs=[(int(rows[i]),int(cols[j])) for i,j in zip(a,b) if np.isfinite(corr[rows[i],cols[j]])]
    complete=len(pairs)==corr.shape[0]==corr.shape[1]
    return dict(max_similarity=float(np.nanmax(corr[:,cols],axis=0).mean()) if len(cols) else None,
        matched_similarity=float(np.mean([corr[i,j] for i,j in pairs])) if complete else None,
        matching_complete=complete,pairs=pairs,n_defined_domains=len(cols))


def metric_neighbors(xy,sections,k,dest):
    nn=mmap_create(dest,(len(xy),k),np.int32);nn[:]=-1
    for s in np.unique(sections):
        ids=np.flatnonzero(sections==s);tree=cKDTree(xy[ids]);count=min(len(ids),k+1)
        for a,b in chunks(len(ids),8192):
            _,neighbors=tree.query(xy[ids[a:b]],k=count,workers=1);neighbors=np.asarray(neighbors).reshape(b-a,count)
            for offset,near in enumerate(neighbors):
                local=a+offset;near=near[near!=local][:k];nn[ids[local],:len(near)]=ids[near]
    nn.flush();return nn


def metrics(run):
    cfg=configuration(run);run=Path(run);out=run/'analysis/metrics';out.mkdir(exist_ok=True);ids=np.load(run/'analysis/evaluation_ids.npy');xy=array(run,'xy_um')[ids];sec=array(run,'sections')[ids]
    nn=metric_neighbors(xy,sec,cfg['metric_neighbors'],out/'neighbors.npy');refs=read_json(run/'analysis/references.json');vocab=read_json(run/'prepared/vocabulary.json');types=array(run,'types')[ids];fine=array(run,'cell_cluster')[ids];rows=[];domain_rows=[];summaries={}
    for name,info in refs.items():
        k=info['k'];reference=np.load(run/f'analysis/reference_{name}.npy');score,per,ref_sizes=homogeneity(reference,nn,k);threshold=float(np.nanquantile(per,.2));comp,_=composition(reference,types,k,len(vocab['cell_subclass']))
        fcomp,_=composition(reference,fine,k,len(vocab['cell_cluster']));pd.DataFrame(fcomp,index=info['names'],columns=vocab['cell_cluster']).to_csv(out/f'{name}_reference_cluster_composition.tsv',sep='\t')
        pd.DataFrame(comp,index=info['names'],columns=vocab['cell_subclass']).to_csv(out/f'{name}_reference_subclass_composition.tsv',sep='\t')
        domain_rows.extend(dict(reference=name,method='reference',k=k,domain=i,reference_id=info['ids'][i],reference_name=info['names'][i],n_cells=int(ref_sizes[i]),spatial_similarity=finite(per[i]),threshold=threshold) for i in range(k))
        summaries[name]=dict(k=k,threshold=threshold,n_cells=len(ids),spatial_similarity=score,discrete_fraction=float(np.mean(per[np.isfinite(per)]>=threshold)))
        rows.append(dict(reference=name,k=k,method='reference',method_name='CCF (Allen registration)',neighborhood_spatial_similarity=score,proportion_of_discrete_domains=summaries[name]['discrete_fraction'],max_domain_CCF_similarity=1.,matched_domain_CCF_similarity=1.,n_evaluation_cells=len(ids),n_domains_on_evaluation_cells=k))
        for method,display in METHODS.items():
            labels=np.load(run/f'analysis/labels/{method}_k{k}.npy',mmap_mode='r')[ids];require((labels>=0).all(),'Missing labels on common cells')
            mean,ds,counts=homogeneity(labels,nn,k);pc,size=composition(labels,types,k,len(vocab['cell_subclass']));corr=correlations(comp,pc);corr[:,size==0]=np.nan;m=match(corr)
            rows.append(dict(reference=name,k=k,method=method,method_name=display,neighborhood_spatial_similarity=mean,
                proportion_of_discrete_domains=float(np.sum(ds>=threshold)/k),max_domain_CCF_similarity=m['max_similarity'],matched_domain_CCF_similarity=m['matched_similarity'],n_evaluation_cells=len(ids),n_domains_on_evaluation_cells=int((size>0).sum())))
            prefix=f'{name}_{method}';pd.DataFrame(corr,index=info['names'],columns=[f'D{i}' for i in range(k)]).to_csv(out/f'{prefix}_correlations.tsv',sep='\t');json_write(out/f'{prefix}_matching.json',m)
            fc,_=composition(labels,fine,k,len(vocab['cell_cluster']));pd.DataFrame(fc,index=[f'D{i}' for i in range(k)],columns=vocab['cell_cluster']).to_csv(out/f'{prefix}_cluster_composition.tsv',sep='\t')
            pd.DataFrame(pc,index=[f'D{i}' for i in range(k)],columns=vocab['cell_subclass']).to_csv(out/f'{prefix}_subclass_composition.tsv',sep='\t')
            domain_rows.extend(dict(reference=name,method=method,k=k,domain=i,n_cells=int(size[i]),spatial_similarity=finite(ds[i]),threshold=threshold) for i in range(k))
        log(f'Figure 2 analogue metrics: {name}, k={k}')
    pd.DataFrame(rows).to_csv(out/'fig2_metrics.tsv',sep='\t',index=False);pd.DataFrame(domain_rows).to_csv(out/'per_domain.tsv',sep='\t',index=False)
    json_write(out/'reference_summary.json',summaries);json_write(out/'definitions.json',dict(
        c='Mean cell-level fraction of up to 100 closest same-section evaluation neighbors with the same domain label; self excluded',
        d='Fraction of requested domains with mean cell-level score at least the reference-domain 20th percentile; missing domains count as failures',
        e='Mean over model domains of maximum Pearson r between subclass composition vectors and reference vectors; undefined domains disclosed',
        f='Maximum-sum one-to-one Pearson matching; average reported only for complete equal-sized matching',
        g='Reference-by-model Pearson correlation of cell-subclass composition; r > 0.9 highlighted',
        hi='Cell-cluster composition, row-normalized to sum to 1',
        reference='20231215 Allen parcellation_division/structure/substructure assignments: 25/354/670 labels, including explicit unassigned/residual categories; no ontology-depth coarsening',
        evaluation='Identical eligible reference-labelled cells for both CT2 variants; reference excluded from model fitting; all-cell exploratory analysis, not independent-mouse validation'))
    log('ANATOMICAL COMPARISON METRICS: PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['smooth','metrics'],required=True);a=p.parse_args();globals()[a.stage](a.run)
