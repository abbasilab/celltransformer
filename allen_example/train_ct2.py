"""Single-GPU launcher around the UNMODIFIED CT2 LightningModule."""
import argparse
import csv
import importlib.metadata
import platform
import torch
import lightning as L
from omegaconf import OmegaConf
from common import *
from data_ct2 import CachedDataset,make_loader,audit_sampler
from celltransformer.train.main import SpatialTransformerModule,zinb_loss_log2
from celltransformer.distributions import ZeroInflatedNegativeBinomial
from celltransformer.extract.main import _load_model

class DataModule(L.LightningDataModule):
    def __init__(self,run):super().__init__();self.run=Path(run)
    def train_dataloader(self):
        return make_loader(self.run,array(self.run,'train_ids'),epoch=self.trainer.current_epoch,shuffle=True)
    def val_dataloader(self):return make_loader(self.run,array(self.run,'validation_ids'))

def forward(module,items):
    batch=module._prepare_batch(items)
    out=module.model(**{k:v for k,v in batch.items() if k!='gt_expr'})
    loss,_=zinb_loss_log2(ZeroInflatedNegativeBinomial(**out['nb_params']),batch['gt_expr'])
    return loss,out

def model_hash(model,prefix=''):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if name.startswith(prefix):h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def preflight(module,run):
    cfg=configuration(run);require(cfg['device']=='cuda','Production training requires CUDA; CPU fallback is not validated for this repository')
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA with bfloat16 support is required')
    device=torch.device('cuda');module.to(device);module.eval();dataset=CachedDataset(run)
    degree=np.diff(dataset.ptr);ids=np.flatnonzero(array(run,'eligible'))
    check=ids[np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int)]
    sampler=audit_sampler(run,check)
    # Small unequal-length batches exercise the native jagged CUDA path.
    items=[dataset[int(i)] for i in check[:2]]
    items=[v._replace(observed_expression=v.observed_expression[:12],observed_cell_type=v.observed_cell_type[:12],num_cells_obs=min(12,v.num_cells_obs)) for v in items]
    saved_precision=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    with torch.no_grad():
        _,pair=forward(module,items);_,single=forward(module,items[:1])
        torch.testing.assert_close(pair['spatial_context'].values()[:1],single['spatial_context'].values(),rtol=5e-4,atol=5e-4)
        changed=[items[0]._replace(masked_expression=items[0].masked_expression+2)]+items[1:]
        _,other=forward(module,changed)
        torch.testing.assert_close(pair['spatial_context'].values(),other['spatial_context'].values(),rtol=0,atol=0)
        if len(items)>1:
            changed=[items[0],items[1]._replace(observed_expression=items[1].observed_expression+2)]
            _,other=forward(module,changed)
            torch.testing.assert_close(pair['spatial_context'].values()[:1],other['spatial_context'].values()[:1],rtol=5e-4,atol=5e-4)
    torch.set_float32_matmul_precision(saved_precision)
    # Stress the densest neighborhoods up to the configured batch/token budget.
    extreme=ids[np.argsort(degree[ids])[-cfg['batch_size']:][::-1]]
    loader=make_loader(run,extreme);batch=next(iter(loader));del loader
    module.train();module.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=forward(module,batch)
    require(bool(torch.isfinite(loss)),'Nonfinite native CT2 preflight loss');loss.backward()
    require(all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None),'Nonfinite native CT2 gradients')
    for prefix in ['encoder.','decoder.','embedder.','attn_pool.','nb_proj.']:
        require(sum(float(p.grad.abs().sum()) for name,p in module.model.named_parameters() if name.startswith(prefix) and p.grad is not None)>0,'No gradient in '+prefix)
    result=dict(sampler=sampler,center_masking='PASSED',batch_isolation='PASSED',
        bf16_forward_backward='PASSED',core_gradients='PASSED',stress_batch_centers=len(batch),
        stress_batch_tokens=sum(v.num_cells_obs+1 for v in batch),peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
    module.zero_grad(set_to_none=True);module.cpu();torch.cuda.empty_cache()
    json_write(Path(run)/'celltransformer/preflight.json',result)
    log('CT2 GPU / SAMPLER PREFLIGHT: PASSED')

class History(L.Callback):
    def __init__(self,run):self.run=Path(run);self.out=self.run/'celltransformer';self.cfg=configuration(run)
    def on_train_start(self,trainer,module):
        self.start=time.monotonic();self.seen=0;self.epoch_seen=0;self.epoch_sum=0;self.window_sum=0;self.window_n=0
        self.records=read_json(self.out/'epochs.json') if (self.out/'epochs.json').exists() else []
        self.records=[r for r in self.records if r['epoch']<=trainer.current_epoch]
        path=self.out/'train_hist.csv'
        if path.exists():
            # Restart always restores a completed-epoch checkpoint. Remove later logs.
            import pandas as pd
            temp=path.with_suffix('.csv.tmp');first=True
            for part in pd.read_csv(path,chunksize=100000):
                part=part[part['update']<=trainer.global_step];part.to_csv(temp,mode='w' if first else 'a',header=first,index=False);first=False
            temp.replace(path)
        self.stream=open(path,'a',buffering=1,newline='');self.writer=csv.writer(self.stream)
        if self.stream.tell()==0:self.writer.writerow(['update','epoch','loss','learning_rate','batch_centers'])
        log(f'CT2 TRAINING: {self.cfg["epochs"]} full epochs; starting epoch {trainer.current_epoch+1}; restored updates={trainer.global_step:,}')
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        n=sum(v.num_cells_obs>0 for v in batch);loss=float(outputs['loss']);require(np.isfinite(loss),'Nonfinite training loss')
        self.epoch_seen+=n;self.epoch_sum+=loss*n;self.seen+=n;self.window_sum+=loss*n;self.window_n+=n
        lr=trainer.optimizers[0].param_groups[0]['lr'];step=trainer.global_step
        self.writer.writerow([step,trainer.current_epoch+1,loss,lr,n])
        if step%self.cfg['log_every']==0:
            total=len(array(self.run,'train_ids'));rate=self.seen/max(time.monotonic()-self.start,1e-9)
            remaining=(self.cfg['epochs']-trainer.current_epoch)*total-self.epoch_seen
            log(f'Epoch {trainer.current_epoch+1}/{self.cfg["epochs"]}; update {step:,}; centers {self.epoch_seen:,}/{total:,}; '
                f'mean NLL={self.window_sum/self.window_n:.6f}; lr={lr:.4g}; {rate:.1f} centers/s; training ETA={remaining/rate/3600:.1f} h')
            self.window_sum=0;self.window_n=0
            json_write(self.out/'progress.json',dict(epoch=trainer.current_epoch+1,completed_updates=step,epoch_centers=self.epoch_seen,training_finished=False))
    def on_train_epoch_end(self,trainer,module):
        require(self.epoch_seen==len(array(self.run,'train_ids')),'Epoch did not cover every training center exactly once')
        value=trainer.callback_metrics.get('valid/nll_epoch')
        record=dict(epoch=trainer.current_epoch+1,training_nll=self.epoch_sum/self.epoch_seen,
            validation_nll=float(value) if value is not None else None,centers=self.epoch_seen,
            optimizer_updates=trainer.global_step,learning_rate=trainer.optimizers[0].param_groups[0]['lr'])
        self.records.append(record);json_write(self.out/'epochs.json',self.records)
        self.stream.flush();log('Epoch saved: '+json.dumps(record));self.epoch_seen=0;self.epoch_sum=0
    def on_fit_end(self,trainer,module):
        if hasattr(self,'stream'):self.stream.close()
    def on_exception(self,trainer,module,exception):
        if hasattr(self,'stream'):self.stream.close()

def export(run,checkpoint):
    cfg=configuration(run);out=Path(run)/'celltransformer';hash_=digest(checkpoint)
    torch.set_float32_matmul_precision('highest')
    model=_load_model(str(checkpoint),cfg['device']);ds=CachedDataset(run);eligible=np.flatnonzero(array(run,'eligible'))
    path=out/'embeddings.npy';progress=out/'export_progress.json';cursor=0
    if progress.exists():
        old=read_json(progress);require(old['checkpoint_sha256']==hash_,'Export checkpoint changed');cursor=old['centers_exported']
        emb=np.lib.format.open_memmap(path,mode='r+')
    else:
        emb=mmap_create(path,(len(ds),model.attn_pool.output_dim),np.float32)
        for a,b in chunks(len(ds)):emb[a:b]=np.nan
    wrapper=SpatialTransformerModule(OmegaConf.create(cfg['model']),OmegaConf.create(cfg['optimization']))
    wrapper.model=model;wrapper.to(cfg['device'])
    for batch in make_loader(run,eligible[cursor:]):
        with torch.no_grad():
            fields=wrapper._prepare_batch(batch)
            result=model(**{k:v for k,v in fields.items() if k!='gt_expr'})
        values=result['spatial_context'].values().float().cpu().numpy()
        require(np.isfinite(values).all(),'Nonfinite CT2 embeddings')
        end=cursor+len(batch);emb[eligible[cursor:end]]=values
        if end//16000!=cursor//16000 or end==len(eligible):
            emb.flush();json_write(progress,dict(centers_exported=end,total_centers=len(eligible),checkpoint_sha256=hash_))
            log(f'CT2 embedding export: {end:,}/{len(eligible):,}')
        cursor=end
    require(cursor==len(eligible),'Incomplete embedding export');emb.flush()
    return len(eligible),hash_

def main(run,preflight_only=False):
    run=Path(run);cfg=configuration(run);out=run/'celltransformer';out.mkdir(exist_ok=True)
    L.seed_everything(cfg['seed'],workers=True);torch.set_num_threads(cfg['threads'])
    require(not cfg['software_test'],'Full model execution is GPU-only; use adapter tests on a CPU')
    module=SpatialTransformerModule(OmegaConf.create(cfg['model']),OmegaConf.create(cfg['optimization']))
    require(all(p.requires_grad for p in module.parameters()),'A model parameter is frozen')
    initial=model_hash(module.model);encoder_initial=model_hash(module.model,'encoder.')
    vocabulary=read_json(run/'prepared/vocabulary.json')
    json_write(out/'input_schema.json',dict(
        genes=read_json(run/'prepared/genes.json'),
        gene_symbols=read_json(run/'prepared/gene_symbols.json'),
        cell_type_labels=vocabulary['cell_subclass'],
        cell_type_mapping={v:i for i,v in enumerate(vocabulary['cell_subclass'])},
        expression_transform=cfg['input_transform'],
        patch_width_um=cfg['patch_width_um'],coordinates='x_reconstructed/y_reconstructed * 1000 (micrometers)',
        taxonomy_release='20231215',model_config=cfg['model'],source_manifest_sha256=cfg['source_manifest_sha256']))
    preflight(module,run)
    if preflight_only:return
    # Preflight backward changes no weights; reset RNG to make fresh runs reproducible.
    L.seed_everything(cfg['seed'],workers=True)
    environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        module_file=__import__('inspect').getfile(SpatialTransformerModule),
        versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    json_write(run/'environment.json',environment)
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    callback=L.pytorch.callbacks.ModelCheckpoint(dirpath=checkpoints,filename='epoch{epoch:02d}-step{step}',
        auto_insert_metric_name=False,save_top_k=-1,save_last=True,every_n_epochs=1,save_on_train_epoch_end=True)
    history=History(run)
    trainer=L.Trainer(accelerator='gpu',devices=1,strategy='auto',max_epochs=cfg['epochs'],
        precision=cfg['optimization']['precision'],accumulate_grad_batches=1,deterministic=True,
        logger=L.pytorch.loggers.CSVLogger(str(out/'lightning_logs')),callbacks=[history,callback],
        reload_dataloaders_every_n_epochs=1,enable_progress_bar=False,log_every_n_steps=cfg['log_every'],num_sanity_val_steps=2)
    checkpoint=checkpoints/'last.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.exists() else None
    if saved is not None and int(saved['epoch'])+1>=cfg['epochs']:
        module.load_state_dict(saved['state_dict']);completed_updates=int(saved['global_step'])
        log('Final epoch checkpoint already complete; continuing export without rewriting it')
    else:
        trainer.fit(module,datamodule=DataModule(run),ckpt_path=str(checkpoint) if checkpoint.exists() else None)
        completed_updates=trainer.global_step
    del saved
    records=read_json(out/'epochs.json');require(len(records)==cfg['epochs'],'Requested epoch budget incomplete')
    require(model_hash(module.model,'encoder.')!=encoder_initial,'Encoder weights did not change')
    module.cpu();del module;torch.cuda.empty_cache();n_exported,hash_=export(run,checkpoint)
    summary=dict(model='CellTransformer2 / SpatialCellTransformer',requested_epochs=cfg['epochs'],
        completed_updates=completed_updates,n_exported=n_exported,initial_full_validation_nll=None,
        final_full_validation_nll=records[-1]['validation_nll'],exported_checkpoint=str(checkpoint),checkpoint_sha256=hash_,
        initialization='random',pretrained_checkpoint_loaded=False,all_parameters_trainable=True,initial_sha256=initial,
        source_manifest_sha256=digest(HERE/'CT2_SOURCE_MANIFEST.json'),software_test=False,
        export_precision='float32 forward and stored embeddings, as in native extract')
    json_write(out/'summary.json',summary)
    json_write(out/'progress.json',dict(completed_epochs=cfg['epochs'],completed_updates=completed_updates,training_finished=True))
    log('CT2 TRAINING AND FINAL-CHECKPOINT EXPORT: PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();main(a.run,a.preflight_only)
