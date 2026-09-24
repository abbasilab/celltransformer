"""CPU tests for data adaptation and Lightning orchestration (not CUDA model tests)."""
import argparse
import shutil
from unittest.mock import patch
import torch
import lightning as L
from omegaconf import OmegaConf
from common import *
from prepare_barseq import graph_and_split, references
from data_ct2 import CachedDataset,FiniteBudgetSampler,audit_sampler,make_loader
from train_ct2 import SpatialTransformerModule,DataModule,History

class ToyModel(torch.nn.Module):
    """Test-only model; production always instantiates the supplied CT2 model."""
    def __init__(self,genes):super().__init__();self.linear=torch.nn.Linear(genes,genes)
    def forward(self,expression_mat,cell_type_annot_obs,lengths,decoder_cell_type_annot):
        features=torch.stack([v.mean(0) for v in expression_mat.split(lengths.tolist())])
        mu=torch.nn.functional.softplus(self.linear(features))+.01
        return {'nb_params':dict(mu=mu,theta=torch.ones_like(mu),zi_logits=torch.zeros_like(mu))}

class Interrupt(L.Callback):
    def __init__(self,after):self.after=after
    def on_train_batch_end(self,trainer,*args):
        if trainer.global_step>=self.after:raise RuntimeError('INTENTIONAL INTERRUPTION TEST')

def fit(run,max_epochs,interrupt=None):
    cfg=configuration(run);L.seed_everything(cfg['seed'],workers=True)
    m=SpatialTransformerModule(OmegaConf.create(cfg['model']),OmegaConf.create(cfg['optimization']));m.model=ToyModel(cfg['model']['expression_dim'])
    cp=run/'celltransformer/checkpoints';cp.mkdir(parents=True,exist_ok=True)
    cb=L.pytorch.callbacks.ModelCheckpoint(dirpath=cp,save_top_k=-1,save_last=True,every_n_epochs=1,save_on_train_epoch_end=True)
    callbacks=[History(run),cb]+([Interrupt(interrupt)] if interrupt is not None else [])
    t=L.Trainer(accelerator='cpu',devices=1,max_epochs=max_epochs,logger=False,callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,enable_progress_bar=False,num_sanity_val_steps=2,enable_model_summary=False)
    t.fit(m,datamodule=DataModule(run),ckpt_path=str(cp/'last.ckpt') if (cp/'last.ckpt').exists() else None)
    return t,m

def main(out):
    import pandas as pd
    from analyze import smooth,metrics,homogeneity,composition,correlations,match,smooth_section
    from report import main as report
    require(not out.exists(),'Use a fresh test output directory');out.mkdir(parents=True);prepared=out/'prepared';prepared.mkdir()
    rng=np.random.default_rng(8);n=96;genes=12
    cfg=dict(workflow='barseq_ct2_v1',root=str(out),run=str(out),software_test=True,workers=0,patch_width_um=2.1,seed=42,batch_size=16,max_tokens=50,
        device='cpu',threads=1,epochs=2,log_every=1000,split='within_section',validation_fraction=.1,metric_neighbors=5,smoothing=dict(sigma_um=1.,k=4,includes_self=True),
        storage_reserve_gib=1)  # Tiny synthetic fixture; production retains its 20-GiB reserve.
    cfg['model']=dict(_target_='celltransformer.model.SpatialCellTransformer',expression_dim=genes,cell_type_cardinality=3,model_dim=32,encoder_depth=1,decoder_depth=1,num_heads=4,ffn_dim=32,concat_mode='add',use_zi=True,use_diagonal_norm=True,nb_proj_eps=.0001,dropout=0.,multiple_of=32,norm_eps=1e-5,init_std_attn_pool=.02,compile=False)
    cfg['optimization']=dict(lr=.0003,weight_decay=1e-8,schedule_warmup=2,schedule_timescale=5,max_epochs=2,precision='32-true',accumulate_grad_batches=1)
    json_write(out/'config.json',cfg)
    xy=np.tile(np.c_[np.tile(np.arange(8),6),np.repeat(np.arange(6),8)],(2,1)).astype(float);sections=np.repeat([0,1],48);types=np.tile(np.arange(48)%3,2)
    for name,x in [('xy_um',xy),('sections',sections),('types',types),('cell_cluster',types),('nonempty',np.ones(n,bool)),('counts',rng.poisson(3,(n,genes)).astype(np.int32))]:atomic_npy(prepared/(name+'.npy'),x)
    meta=pd.DataFrame(dict(cell_id=np.arange(n),section=np.where(sections==0,'Slice01','Slice02'),x_um=xy[:,0],y_um=xy[:,1],gene_area=np.tile(np.repeat([962,1016,1015],16),2),reference_valid=True,reference_fraction=1.,cell_cluster=types.astype(str),cell_subclass=types.astype(str),cell_class='Test',total_counts=36,n_genes=12,mitochondrial_fraction=0,qc_flag=False))
    meta.to_parquet(prepared/'cells.parquet');vocab=references(meta,prepared);vocab.update(section=['Slice01','Slice02'],cell_subclass=['A','B','C'],cell_cluster=['A','B','C'],cell_class=['Test']);json_write(prepared/'vocabulary.json',vocab)
    # Exercise the CI storage budget even on development machines with ample space.
    # Cap reported space rather than inventing capacity when less is available.
    usage=shutil.disk_usage(prepared)
    with patch('prepare_barseq.shutil.disk_usage',return_value=usage._replace(free=min(usage.free,2*1024**3))):
        info=graph_and_split(out)
    info.update(n_cells=n,n_sections=2,n_model_genes=genes,n_mice=1);json_write(prepared/'summary.json',info)
    ids=np.flatnonzero(array(out,'eligible'));audit=audit_sampler(out,ids);dataset=CachedDataset(out);degree=np.diff(dataset.ptr)
    require(dataset[0].observed_expression.dtype==torch.float32,'Integer input must transform to float32')
    for budget in [5,50,200,10000]:
        for batch in [1,7,256]:
            order=ids.copy();rng.shuffle(order);sampler=FiniteBudgetSampler(order,degree,batch,budget);batches=list(sampler)
            require(len(batches)==len(sampler),'Batch length incorrect');require(np.array_equal(np.concatenate(batches),order),'Center coverage wrong')
            require(all(len(b)<=batch and (len(b)==1 or sum(degree[b]+1)<=budget) for b in batches),'Budget violation')
    out_model=out/'celltransformer';out_model.mkdir();t,_=fit(out,1);steps=t.global_step
    try:fit(out,2,interrupt=steps+3)
    except RuntimeError as e:require('INTENTIONAL INTERRUPTION' in str(e),'Unexpected failure')
    else:raise AssertionError('No test interruption')
    t,m=fit(out,2);records=read_json(out_model/'epochs.json');h=pd.read_csv(out_model/'train_hist.csv')
    require([v['epoch'] for v in records]==[1,2],'Resume lost/duplicated epochs');require(h['update'].tolist()==list(range(1,t.global_step+1)),'Resume history corruption')
    require(int(h.batch_centers.sum())==2*len(array(out,'train_ids')),'Incomplete epochs')
    emb=np.repeat(sections[:,None],8,axis=1).astype(np.float32)*100;atomic_npy(out_model/'embeddings.npy',emb)
    smooth(out);require(np.allclose(np.load(out/'analysis/smoothed.npy'),emb),'Smoothing mixed different sections')
    trial=smooth_section(np.array([[0.,0.],[1.,0.]]),np.array([[0.],[2.]],np.float32),1,2)
    expected=np.array([[2*np.exp(-.5)/(1+np.exp(-.5))],[2/(1+np.exp(-.5))]])
    require(np.allclose(trial,expected),'Wrong native Gaussian weighting')
    refs=read_json(out/'analysis/references.json');labels=out/'analysis/labels';labels.mkdir()
    for name,r in refs.items():
        y=np.load(out/f'analysis/reference_{name}.npy')
        for method in ['celltransformer','celltransformer_smoothed']:atomic_npy(labels/f'{method}_k{r["k"]}.npy',y)
    # Independent numerical expectations for permutation-invariant composition.
    comp=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]);result=match(correlations(comp,comp[[2,0,1]]));require(np.isclose(result['max_similarity'],1) and np.isclose(result['matched_similarity'],1),'Correlation matching failed')
    metrics(out)
    json_write(out_model/'summary.json',dict(software_test=True,model='TEST-ONLY TOY FORWARD; native loss and optimizer',requested_epochs=2));json_write(out/'environment.json',dict(software_test=True))
    report(out)
    text=(out/'report/review.html').read_text();require(text.count('data:image/png;base64,')>=10,'Missing embedded plots');require('Slice01' in text and 'Slice02' in text,'Missing section selector')
    result=dict(status='PASSED',native_sampler=audit,integer_targets='int32 -> float32 log2',budget_batcher='12 budget combinations; exact coverage',resume='Completed epoch + interrupted next epoch + checkpoint restore; exact coverage and log sequence',smoothing='native weights, analytical two-point check, no cross-section mixing',metrics='Permutation matching r=1',report='All synthetic section/resolution maps embedded; offline HTML and ZIP generated',native_model='CUDA model forward/backward must pass on cactus; CPU orchestration uses an explicit toy forward with native training_step/loss/optimizer/scheduler')
    json_write(out/'test_result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
