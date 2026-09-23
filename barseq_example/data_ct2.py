"""Finite epochs with the unchanged CT2 BudgetBatcher and cached CT2 neighborhoods."""
from types import SimpleNamespace
import tempfile
import torch
from torchdata.nodes import BaseNode
from common import *
from celltransformer.loader.types import NeighborhoodMetadata
from celltransformer.loader.sampler import BudgetBatcher

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self,run):
        self.run=Path(run);self.counts=array(run,'counts');self.types=array(run,'types')
        self.sections=array(run,'sections');self.ptr=array(run,'neighbor_indptr');self.edges=array(run,'neighbor_indices')
    def __len__(self):return len(self.counts)
    def __getitem__(self,i):
        i=int(i);near=self.edges[self.ptr[i]:self.ptr[i+1]]
        return NeighborhoodMetadata(torch.from_numpy(np.log2(np.asarray(self.counts[near],dtype=np.float32)+1)),
            torch.from_numpy(np.log2(np.asarray(self.counts[i:i+1],dtype=np.float32)+1)),
            torch.from_numpy(np.array(self.types[near],dtype=np.int64)),torch.tensor([int(self.types[i])]),
            len(near),None,int(self.sections[i]))

class DegreeNode(BaseNode):
    def __init__(self,ids,degrees):super().__init__();self.ids=ids;self.degrees=degrees;self.i=0
    def reset(self,initial_state=None):super().reset(initial_state);self.i=0
    def get_state(self):return {'i':self.i}
    def next(self):
        if self.i==len(self.ids):raise StopIteration
        i=int(self.ids[self.i]);self.i+=1
        return SimpleNamespace(index=i,num_cells_obs=int(self.degrees[i]))

class FiniteBudgetSampler:
    def __init__(self,ids,degrees,batch_size,max_tokens):
        self.ids=ids;self.degrees=degrees;self.batch_size=batch_size;self.max_tokens=max_tokens
        # Compute the exact finite batch count (including the short last batch).
        n=tokens=groups=0
        for cost in np.asarray(degrees[ids])+1:
            cost=int(cost)
            if n and (n>=batch_size or tokens+cost>max_tokens):groups+=1;n=tokens=0
            n+=1;tokens+=cost
        self.length=groups+bool(n)
    def __len__(self):return self.length
    def __iter__(self):
        node=BudgetBatcher(DegreeNode(self.ids,self.degrees),max_batch_size=self.batch_size,max_tokens=self.max_tokens)
        node.reset()
        while True:
            try:yield [item.index for item in node.next()]
            except StopIteration:return

def identity(batch):return batch

def make_loader(run,ids,epoch=0,shuffle=False):
    cfg=configuration(run);ids=np.asarray(ids).copy()
    if shuffle:np.random.default_rng(cfg['seed']+epoch).shuffle(ids)
    dataset=CachedDataset(run);degree=np.diff(dataset.ptr)
    sampler=FiniteBudgetSampler(ids,degree,cfg['batch_size'],cfg['max_tokens'])
    kw=dict(dataset=dataset,batch_sampler=sampler,collate_fn=identity,num_workers=cfg['workers'],
        generator=torch.Generator().manual_seed(cfg['seed']+epoch),pin_memory=cfg['device']=='cuda')
    if cfg['workers']:kw.update(prefetch_factor=2,persistent_workers=False)
    return torch.utils.data.DataLoader(**kw)

def audit_sampler(run,ids):
    import anndata as ad
    import pandas as pd
    from celltransformer.loader import CenterMaskNodeOptimized
    cfg=configuration(run);ds=CachedDataset(run);xy=array(run,'xy_um');sec=array(run,'sections');nonempty=array(run,'nonempty')
    half=cfg['patch_width_um']/2
    with tempfile.TemporaryDirectory(prefix='ct2-sampler-',dir=run) as tmp:
        tmp=Path(tmp)
        for i in ids:
            section=np.flatnonzero((sec==sec[i])&nonempty);coords=xy[section]
            rows=section[np.all((coords>=xy[i]-half)&(coords<=xy[i]+half),axis=1)];rows=rows[rows!=i]
            rows=rows[np.argsort(xy[rows,0],kind='stable')]
            actual=ds.edges[ds.ptr[i]:ds.ptr[i+1]]
            require(np.array_equal(rows,actual),'CT2 full-section neighbor membership/order differs')
            sample=np.unique(np.r_[rows,i,section[::max(1,len(section)//20)]])
            adata=ad.AnnData(X=np.log2(np.asarray(ds.counts[sample],dtype=np.float32)+1))
            meta=pd.DataFrame(dict(x=xy[sample,0],y=xy[sample,1],section=sec[sample],cell_type=np.asarray(ds.types[sample],dtype=np.int64)))
            adata.write_h5ad(tmp/'check.h5ad');meta.to_parquet(tmp/'check.parquet',index=False)
            native=CenterMaskNodeOptimized(str(tmp/'check.parquet'),str(tmp/'check.h5ad'),'cell_type',
                (cfg['patch_width_um'],)*2,tissue_section_colname='section')
            ref=native[int(np.flatnonzero(sample==i)[0])];got=ds[int(i)]
            for name in ['observed_expression','masked_expression','observed_cell_type','masked_cell_type']:
                torch.testing.assert_close(getattr(ref,name),getattr(got,name),rtol=0,atol=0)
    return dict(centers_checked=len(ids),full_section_membership='PASSED',native_CT2_tensors='bitwise equal',
        context_filter='nonempty cells, retained BARseq cells with measured expression')
