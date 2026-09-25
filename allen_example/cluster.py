"""Native CT2 single-GPU cuML KMeans, n_init=5, all exported centers."""
import argparse
from common import *
from celltransformer.cluster.main import _fit_single_gpu

def main(run):
    run=Path(run);cfg=configuration(run);out=run/'analysis';dest=out/'labels';dest.mkdir(exist_ok=True)
    refs=read_json(out/'references.json');ks=sorted(set(v['k'] for v in refs.values()));eligible=np.flatnonzero(array(run,'eligible'));records=[]
    for method,path in [('celltransformer',run/'celltransformer/embeddings.npy'),('celltransformer_smoothed',out/'smoothed.npy')]:
        x=np.asarray(np.load(path,mmap_mode='r')[eligible],dtype=np.float32)
        for a,b in chunks(len(x)):require(np.isfinite(x[a:b]).all(),'Nonfinite clustering input')
        for k in ks:
            target=dest/f'{method}_k{k}.npy'
            if target.exists():
                labels=np.load(target);require(len(labels)==len(array(run,'eligible')),'Incomplete label export')
            else:
                log(f'{method}: native CT2 KMeans k={k}, n={len(x):,}, n_init=5')
                y=_fit_single_gpu(x,k,cfg['seed']);labels=np.full(len(array(run,'eligible')),-1,np.int32);labels[eligible]=y;atomic_npy(target,labels)
            count=np.bincount(labels[eligible],minlength=k);require(len(count)==k and (count>0).all(),f'{method} k={k}: empty domains; no relabelling to hide collapse')
            records.append(dict(method=method,k=k,n_cells=len(eligible),n_init=5,seed=cfg['seed'],populations=count.tolist()))
        del x
    json_write(out/'clustering.json',records);log('CT2 CLUSTERING: PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
