"""Resumable Allen 1 pipeline; standard-library launcher works before environment setup."""
import argparse, datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
from source_layout import audit_example, audit_source, copy_source

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def lock_root(root):
    handle=open(root/'ALLEN_RUN.lock','a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A Allen 1 pipeline is already running under this storage root; follow its logs instead of launching another copy')
    return handle

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--python',type=Path);p.add_argument('--rapids-python',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--minimum-free-gib',type=int,default=80)
    args=p.parse_args()
    if args.epochs<=0:p.error('--epochs must be positive')
    if args.resume:
        run=args.resume.resolve();cfg=json.loads((run/'config.json').read_text());root=Path(cfg['root']);root_lock=lock_root(root);snapshot=run/'runner_snapshot'
    else:
        if args.root is None:p.error('--root is required; select a writable filesystem with at least 80 GiB free')
        root=args.root.resolve();root.mkdir(parents=True,exist_ok=True);root_lock=lock_root(root)
        free=shutil.disk_usage(root).free/1024**3
        if free<args.minimum_free_gib:raise SystemExit(f'{root}: only {free:.1f} GiB free; need {args.minimum_free_gib} GiB. No download or training started.')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}'
        run=root/'runs'/stamp;run.mkdir(parents=True);snapshot=run/'runner_snapshot'
        audit_example();audit_source()
        shutil.copytree(HERE,snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store','results','docs'))
        copy_source(snapshot)
        rapids=args.rapids_python or root/'envs/rapids-24.12/bin/python'
        if args.rapids_python and not rapids.is_file():p.error('--rapids-python does not exist')
        if args.python and not args.python.is_file():p.error('--python does not exist')
        cfg=dict(workflow='allen1_ct2_v1',root=str(root),run=str(run),python=str((args.python or root/'envs/celltransformer2-v1/bin/python').resolve()),rapids_python=str(rapids.resolve()),
            epochs=args.epochs,n_genes=500,batch_size=256,max_tokens=16384,patch_width_um=85.0,workers=0,seed=42,threads=8,device='cuda',
            storage_reserve_gib=20,split='random_centers',validation_fraction=.2,log_every=100,software_test=False,
            model=dict(_target_='celltransformer.model.SpatialCellTransformer',expression_dim=500,cell_type_cardinality=384,model_dim=384,encoder_depth=4,decoder_depth=4,num_heads=8,ffn_dim=384,concat_mode='add',use_zi=True,use_diagonal_norm=True,nb_proj_eps=1e-4,dropout=0.0,multiple_of=256,norm_eps=1e-5,init_std_attn_pool=.02,compile=False),
            optimization=dict(lr=.0003,weight_decay=1e-8,schedule_warmup=500,schedule_timescale=5000,accumulate_grad_batches=1,max_epochs=args.epochs,precision='bf16-mixed'),
            smoothing=dict(sigma_um=12.01,k=500,includes_self=True),metric_neighbors=100,
            initialization='random; all weights trained',epoch_definition='Every eligible training center exactly once; incomplete final batch retained',
            export_checkpoint='last completed epoch, not best validation checkpoint',
            adaptations=['Single GPU, finite full epochs, cached native-equivalent neighborhoods, 500 biological MERFISH genes; 50 Blank probes excluded; paper-era taxonomy/CCF, local logging, section-safe memory-bounded smoothing'],
            input_transform='log2(raw molecule counts + 1)',reference_use='Evaluation only; never input as training label',
            source_manifest_sha256=sha(snapshot/'CT2_SOURCE_MANIFEST.json'),
            example_distribution='celltransformer2-test/allen_example',
            reuse_python=bool(args.python),reuse_rapids=bool(args.rapids_python),
            execution_environment={key:os.environ.get(key) for key in ['CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES']})
        write(run/'config.json',cfg);(root/'LATEST_ALLEN_CT2').write_text(str(run)+'\n')
    (root/'logs').mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True);(run/'stage_manifests').mkdir(exist_ok=True)
    lock=open(run/'RUN.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(run/'PID').write_text(str(os.getpid())+'\n')
    # Audit package/metadata and all vendored files before any stage imports.
    package_manifest=snapshot/'PACKAGE_MANIFEST.json'
    if package_manifest.exists():
        for name,expected in json.loads(package_manifest.read_text()).items():
            if sha(snapshot/name)!=expected:raise ValueError('Changed Allen 1 package file: '+name)
    # Audit all vendored files before any import or execution.
    for name,expected in json.loads((snapshot/'CT2_SOURCE_MANIFEST.json').read_text()).items():
        if sha(snapshot/'celltransformer2'/name)!=expected:raise ValueError('Changed CT2 source: '+name)
    if (run/'SUCCESS').exists():(run/'SUCCESS').unlink()
    status_path=run/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else dict(stages={})
    status.update(overall='RUNNING',run=str(run));status.pop('error',None);write(status_path,status)
    (root/'tmp').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(TMPDIR=str(root/'tmp'),ALLEN_ROOT=str(root),ALLEN_PYTHON=cfg['python'],ALLEN_RAPIDS_PYTHON=cfg['rapids_python'],ALLEN_REUSE_PYTHON=str(int(cfg['reuse_python'])),ALLEN_REUSE_RAPIDS=str(int(cfg['reuse_rapids'])),MPLCONFIGDIR=str(root/'matplotlib-cache'))
    child=None
    def terminate(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise KeyboardInterrupt('Pipeline stopped; existing data and completed epochs preserved')
    signal.signal(signal.SIGTERM,terminate)
    def stage(name,cmd,artifacts):
        nonlocal child
        mark=run/'stage_manifests'/f'{name}.json'
        if mark.exists() and name not in ['environment','download']:
            old=json.loads(mark.read_text())
            if all((run/f).is_file() and (run/f).stat().st_size==size for f,size in old.items()):
                print(f'Skipping completed {name}',flush=True);return
            raise ValueError(f'Completed stage {name} has missing/changed-size artifacts; inspect before rerun')
        logfile=run/'logs'/f'{name}.log';status['active_stage']=name;status['stages'][name]=dict(status='RUNNING',log=str(logfile));write(status_path,status)
        print(f'Starting {name}: {logfile}',flush=True)
        reserve=cfg.get('storage_reserve_gib',20)
        if shutil.disk_usage(root).free<reserve*1024**3:raise RuntimeError(f'Less than {reserve} GiB free before {name}; completed data preserved')
        with open(logfile,'a',buffering=1) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);code=child.wait();child=None
        if code:raise RuntimeError(f'{name} failed ({code}); see {logfile}')
        paths=[]
        for pattern in artifacts:
            found=[f for f in run.glob(pattern) if f.is_file()]
            if not found:raise ValueError(f'{name}: required output pattern is missing: {pattern}')
            paths.extend(found)
        if artifacts and not paths:raise ValueError(f'{name}: expected outputs are missing')
        write(mark,{str(f.relative_to(run)):f.stat().st_size for f in paths});status['stages'][name]['status']='PASSED';write(status_path,status)
    try:
        print('ALLEN RUN: '+str(run),flush=True)
        # Always recheck GPU availability even on resume.
        stage('environment',['bash',str(snapshot/'setup_environment.sh')],[])
        py=cfg['python'];rp=cfg['rapids_python']
        stage('download',['python3','-u',str(snapshot/'download.py'),'--root',str(root)],[])
        stage('prepare',[py,'-u',str(snapshot/'prepare_allen.py'),'--run',str(run)],['prepared/*'])
        if args.prepare_only:
            status.update(overall='PREPARED',active_stage=None);write(status_path,status);print('Preparation complete; resume this run to train.',flush=True);return
        stage('train',[py,'-u',str(snapshot/'train_ct2.py'),'--run',str(run)],['celltransformer/summary.json','celltransformer/embeddings.npy','celltransformer/checkpoints/last.ckpt','celltransformer/epochs.json'])
        stage('smooth',[py,'-u',str(snapshot/'analyze.py'),'--run',str(run),'--stage','smooth'],['analysis/smoothed.npy'])
        stage('cluster',[rp,'-u',str(snapshot/'cluster.py'),'--run',str(run)],['analysis/labels/*.npy','analysis/clustering.json'])
        stage('metrics',[py,'-u',str(snapshot/'analyze.py'),'--run',str(run),'--stage','metrics'],['analysis/metrics/*'])
        stage('report',[py,'-u',str(snapshot/'report.py'),'--run',str(run)],['report/review.html','report/results.zip'])
        status.update(overall='PASSED',active_stage=None);write(status_path,status);(run/'SUCCESS').write_text(str(run/'report/review.html')+'\n')
        print('ALLEN CELLTRANSFORMER2 RUN COMPLETED: '+str(run/'report/review.html'),flush=True)
    except BaseException as e:
        status.update(overall='STOPPED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=str(e))
        active=status.get('active_stage')
        if active:status['stages'][active]['status']=status['overall']
        write(status_path,status);raise

if __name__=='__main__':main()
