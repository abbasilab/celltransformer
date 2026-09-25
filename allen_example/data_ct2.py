"""Finite epochs with the unchanged CT2 BudgetBatcher and cached CT2 neighborhoods."""
from types import SimpleNamespace
import tempfile
import torch
from torchdata.nodes import BaseNode
from common import *
from celltransformer.loader.types import NeighborhoodMetadata
from celltransformer.loader.sampler import BudgetBatcher


class CachedDataset(torch.utils.data.Dataset):
    """
    Dataset that returns each cell's neighborhood (the cell plus the other
    cells in its patch), read from the arrays written by prepare_allen.py.
    """
    def __init__(self, run):
        """
        Load counts, cell types, sections, and the neighbor graph for a run.
        """

        self.run = Path(run)
        self.counts = array(run, 'counts')
        self.types = array(run, 'types')
        self.sections = array(run, 'sections')
        #cell i's neighbors are edges[ptr[i]:ptr[i+1]]
        self.ptr = array(run, 'neighbor_indptr')
        self.edges = array(run, 'neighbor_indices')
    def __len__(self):
        """
        Number of cells.
        """

        return len(self.counts)
    def __getitem__(self, i):
        """
        Return cell i's neighborhood: log2(count+1) expression and cell types
        for its neighbors (observed) and for cell i itself (masked).
        """

        i = int(i)
        neighbors = self.edges[self.ptr[i]:self.ptr[i + 1]]

        neighbor_expression = np.log2(np.asarray(self.counts[neighbors], dtype=np.float32) + 1)
        center_expression = np.log2(np.asarray(self.counts[i:i + 1], dtype=np.float32) + 1)  #i:i+1 keeps it 2D (1 x genes)
        neighbor_types = np.array(self.types[neighbors], dtype=np.int64)

        return NeighborhoodMetadata(
            observed_expression=torch.from_numpy(neighbor_expression),
            masked_expression=torch.from_numpy(center_expression),
            observed_cell_type=torch.from_numpy(neighbor_types),
            masked_cell_type=torch.tensor([int(self.types[i])]),
            num_cells_obs=len(neighbors),
            measured_gene_mask=None,  #all genes are measured
            section=int(self.sections[i]),
        )

    #Old, claude:
    # def __init__(self,run):
    #     self.run=Path(run);self.counts=array(run,'counts');self.types=array(run,'types')
    #     self.sections=array(run,'sections');self.ptr=array(run,'neighbor_indptr');self.edges=array(run,'neighbor_indices')
    # def __len__(self):return len(self.counts)
    # def __getitem__(self,i):
    #     i=int(i);near=self.edges[self.ptr[i]:self.ptr[i+1]]
    #     return NeighborhoodMetadata(torch.from_numpy(np.log2(np.asarray(self.counts[near],dtype=np.float32)+1)),
    #         torch.from_numpy(np.log2(np.asarray(self.counts[i:i+1],dtype=np.float32)+1)),
    #         torch.from_numpy(np.array(self.types[near],dtype=np.int64)),torch.tensor([int(self.types[i])]),
    #         len(near),None,int(self.sections[i]))


class DegreeNode(BaseNode):
    """
    torchdata node that walks through center IDs in order, yielding each
    cell's index and neighbor count (what BudgetBatcher needs to batch them).
    """
    def __init__(self, ids, degrees):
        """
        Store center IDs and per-cell neighbor counts.
        """

        super().__init__()
        self.ids = ids
        self.degrees = degrees
        self.position = 0  #index into ids of the next center to return

    def reset(self, initial_state=None):
        """
        Restart from the first center.
        """

        super().reset(initial_state)
        self.position = 0

    def get_state(self):
        """
        Return the current position (so torchdata can checkpoint it).
        """

        return {'i': self.position}

    def next(self):
        """
        Return the next center and its neighbor count.
        """

        if self.position == len(self.ids):
            raise StopIteration

        cell = int(self.ids[self.position])
        self.position += 1
        return SimpleNamespace(index=cell, num_cells_obs=int(self.degrees[cell]))

    #Old, claude:
    # def __init__(self,ids,degrees):super().__init__();self.ids=ids;self.degrees=degrees;self.i=0
    # def reset(self,initial_state=None):super().reset(initial_state);self.i=0
    # def get_state(self):return {'i':self.i}
    # def next(self):
    #     if self.i==len(self.ids):raise StopIteration
    #     i=int(self.ids[self.i]);self.i+=1
    #     return SimpleNamespace(index=i,num_cells_obs=int(self.degrees[i]))


class FiniteBudgetSampler:
    """
    Batch sampler that groups centers into batches capped by both batch size
    and a token budget (each center costs its neighbors + itself).
    """
    def __init__(self, ids, degrees, batch_size, max_tokens):
        """
        Store settings and precompute the exact number of batches.
        """

        self.ids = ids
        self.degrees = degrees
        self.batch_size = batch_size
        self.max_tokens = max_tokens

        #simulate the batching to count batches (including the short last batch)
        n_in_batch = 0
        tokens_in_batch = 0
        n_full_batches = 0
        costs = np.asarray(degrees[ids]) + 1  #neighbors + the center itself
        for cost in costs:
            cost = int(cost)
            #close the current batch if it's full or this center would go over budget
            batch_full = n_in_batch >= batch_size or tokens_in_batch + cost > max_tokens
            if n_in_batch > 0 and batch_full:
                n_full_batches += 1
                n_in_batch = 0
                tokens_in_batch = 0
            n_in_batch += 1
            tokens_in_batch += cost

        #+1 for the last, partly filled batch
        self.length = n_full_batches + (1 if n_in_batch > 0 else 0)
    def __len__(self):
        """
        Number of batches per epoch.
        """

        return self.length
    def __iter__(self):
        """
        Yield batches (lists of center indices) using CT2's BudgetBatcher.
        """

        centers = DegreeNode(self.ids, self.degrees)
        node = BudgetBatcher(centers, max_batch_size=self.batch_size, max_tokens=self.max_tokens)
        node.reset()

        while True:
            try:
                batch = node.next()
            except StopIteration:
                return
            yield [item.index for item in batch]

    #Old, claude:
    # def __init__(self,ids,degrees,batch_size,max_tokens):
    #     self.ids=ids;self.degrees=degrees;self.batch_size=batch_size;self.max_tokens=max_tokens
    #     # Compute the exact finite batch count (including the short last batch).
    #     n=tokens=groups=0
    #     for cost in np.asarray(degrees[ids])+1:
    #         cost=int(cost)
    #         if n and (n>=batch_size or tokens+cost>max_tokens):groups+=1;n=tokens=0
    #         n+=1;tokens+=cost
    #     self.length=groups+bool(n)
    # def __len__(self):return self.length
    # def __iter__(self):
    #     node=BudgetBatcher(DegreeNode(self.ids,self.degrees),max_batch_size=self.batch_size,max_tokens=self.max_tokens)
    #     node.reset()
    #     while True:
    #         try:yield [item.index for item in node.next()]
    #         except StopIteration:return


def identity(batch):
    """
    Collate function that returns the batch unchanged (a list of neighborhoods).
    """

    return batch


def make_loader(run, ids, epoch=0, shuffle=False):
    """
    Build a DataLoader over the given center IDs, batched by FiniteBudgetSampler.
    """

    cfg = configuration(run)
    ids = np.asarray(ids).copy()  #copy so shuffling doesn't change the caller's array

    if shuffle:
        #seed + epoch: a different but reproducible order each epoch
        rng = np.random.default_rng(cfg['seed'] + epoch)
        rng.shuffle(ids)

    dataset = CachedDataset(run)
    degree = np.diff(dataset.ptr)  #neighbor count per cell
    sampler = FiniteBudgetSampler(ids, degree, cfg['batch_size'], cfg['max_tokens'])

    loader_kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=identity,
        num_workers=cfg['workers'],
        generator=torch.Generator().manual_seed(cfg['seed'] + epoch),
        pin_memory=cfg['device'] == 'cuda',
    )
    if cfg['workers']:
        #these options are only allowed when num_workers > 0
        loader_kwargs.update(prefetch_factor=2, persistent_workers=False)

    return torch.utils.data.DataLoader(**loader_kwargs)

    #Old, claude:
    # cfg=configuration(run);ids=np.asarray(ids).copy()
    # if shuffle:np.random.default_rng(cfg['seed']+epoch).shuffle(ids)
    # dataset=CachedDataset(run);degree=np.diff(dataset.ptr)
    # sampler=FiniteBudgetSampler(ids,degree,cfg['batch_size'],cfg['max_tokens'])
    # kw=dict(dataset=dataset,batch_sampler=sampler,collate_fn=identity,num_workers=cfg['workers'],
    #     generator=torch.Generator().manual_seed(cfg['seed']+epoch),pin_memory=cfg['device']=='cuda')
    # if cfg['workers']:kw.update(prefetch_factor=2,persistent_workers=False)
    # return torch.utils.data.DataLoader(**kw)


def audit_sampler(run, ids):
    """
    For each given center, check that the cached neighborhood matches both a
    brute-force recomputation and CT2's own loader (CenterMaskNodeOptimized),
    down to identical tensors.
    """

    import anndata as ad
    import pandas as pd
    from celltransformer.loader import CenterMaskNodeOptimized

    cfg = configuration(run)
    dataset = CachedDataset(run)
    xy = array(run, 'xy_um')
    sections = array(run, 'sections')
    nonempty = array(run, 'nonempty')  #loaded but not used below
    half_width = cfg['patch_width_um'] / 2

    with tempfile.TemporaryDirectory(prefix='ct2-sampler-', dir=run) as tmp:
        tmp = Path(tmp)
        for center in ids:
            #brute-force neighbors: same-section cells inside the square patch, sorted by x
            cells_in_section = np.flatnonzero(sections == sections[center])
            coords = xy[cells_in_section]
            in_patch = np.all((coords >= xy[center] - half_width) & (coords <= xy[center] + half_width), axis=1)
            expected = cells_in_section[in_patch]
            expected = expected[expected != center]  #drop the center itself
            expected = expected[np.argsort(xy[expected, 0], kind='stable')]

            #compare to the cached neighbor list
            cached = dataset.edges[dataset.ptr[center]:dataset.ptr[center + 1]]
            if not np.array_equal(expected, cached):
                raise ValueError('CT2 full-section neighbor membership/order differs')

            #small subset for CT2's loader: the neighborhood, the center, and every ~20th cell in the section
            step = max(1, len(cells_in_section) // 20)
            sample = np.unique(np.r_[expected, center, cells_in_section[::step]])

            #write the subset in the file formats CT2's loader reads
            adata = ad.AnnData(X=np.log2(np.asarray(dataset.counts[sample], dtype=np.float32) + 1))
            meta = pd.DataFrame(dict(
                x=xy[sample, 0],
                y=xy[sample, 1],
                section=sections[sample],
                cell_type=np.asarray(dataset.types[sample], dtype=np.int64),
            ))
            adata.write_h5ad(tmp / 'check.h5ad')
            meta.to_parquet(tmp / 'check.parquet', index=False)

            #build the same neighborhood with CT2's loader
            native = CenterMaskNodeOptimized(str(tmp / 'check.parquet'), str(tmp / 'check.h5ad'), 'cell_type',
                                             (cfg['patch_width_um'],) * 2, tissue_section_colname='section')
            position_in_sample = int(np.flatnonzero(sample == center)[0])
            reference = native[position_in_sample]
            ours = dataset[int(center)]

            #tensors must be exactly equal (zero tolerance)
            for name in ['observed_expression', 'masked_expression', 'observed_cell_type', 'masked_cell_type']:
                torch.testing.assert_close(getattr(reference, name), getattr(ours, name), rtol=0, atol=0)

    return dict(
        centers_checked=len(ids),
        full_section_membership='PASSED',
        native_CT2_tensors='bitwise equal',
        context_filter='All annotation/count-intersection cells, including zero-expression cells',
    )

    #Old, claude:
    # import anndata as ad
    # import pandas as pd
    # from celltransformer.loader import CenterMaskNodeOptimized
    # cfg=configuration(run);ds=CachedDataset(run);xy=array(run,'xy_um');sec=array(run,'sections');nonempty=array(run,'nonempty')
    # half=cfg['patch_width_um']/2
    # with tempfile.TemporaryDirectory(prefix='ct2-sampler-',dir=run) as tmp:
    #     tmp=Path(tmp)
    #     for i in ids:
    #         section=np.flatnonzero(sec==sec[i]);coords=xy[section]
    #         rows=section[np.all((coords>=xy[i]-half)&(coords<=xy[i]+half),axis=1)];rows=rows[rows!=i]
    #         rows=rows[np.argsort(xy[rows,0],kind='stable')]
    #         actual=ds.edges[ds.ptr[i]:ds.ptr[i+1]]
    #         require(np.array_equal(rows,actual),'CT2 full-section neighbor membership/order differs')
    #         sample=np.unique(np.r_[rows,i,section[::max(1,len(section)//20)]])
    #         adata=ad.AnnData(X=np.log2(np.asarray(ds.counts[sample],dtype=np.float32)+1))
    #         meta=pd.DataFrame(dict(x=xy[sample,0],y=xy[sample,1],section=sec[sample],cell_type=np.asarray(ds.types[sample],dtype=np.int64)))
    #         adata.write_h5ad(tmp/'check.h5ad');meta.to_parquet(tmp/'check.parquet',index=False)
    #         native=CenterMaskNodeOptimized(str(tmp/'check.parquet'),str(tmp/'check.h5ad'),'cell_type',
    #             (cfg['patch_width_um'],)*2,tissue_section_colname='section')
    #         ref=native[int(np.flatnonzero(sample==i)[0])];got=ds[int(i)]
    #         for name in ['observed_expression','masked_expression','observed_cell_type','masked_cell_type']:
    #             torch.testing.assert_close(getattr(ref,name),getattr(got,name),rtol=0,atol=0)
    # return dict(centers_checked=len(ids),full_section_membership='PASSED',native_CT2_tensors='bitwise equal',
    #     context_filter='All annotation/count-intersection cells, including zero-expression cells')
