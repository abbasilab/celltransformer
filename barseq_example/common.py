from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
from source_layout import source_root
sys.path.insert(0, str(source_root(HERE) / 'src'))
def require(ok, message):
    if not ok: raise ValueError(message)

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S')+' | '+str(message), flush=True)

def read_json(path): return json.loads(Path(path).read_text())

def json_write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');temp.replace(path)

def digest(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_npy(path, x):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:np.save(f,x,allow_pickle=False)
    temp.replace(path)

def mmap_create(path, shape, dtype):
    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

def array(run, name):
    return np.load(Path(run)/'prepared'/(name+'.npy'),mmap_mode='r')

def chunks(n, size=65536):
    for start in range(0,n,size):yield start,min(start+size,n)

def finite(x):return float(x) if np.isfinite(x) else None

def configuration(run):return read_json(Path(run)/'config.json')

def checkpoint_write(path, value):
    import torch
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    torch.save(value,temp);temp.replace(path)

def compact_categories(values, categories):
    present=np.unique(values[values>=0]);lookup=np.full(len(categories),-1,dtype=np.int16)
    lookup[present]=np.arange(len(present),dtype=np.int16)
    out=np.full(len(values),-1,dtype=np.int16);keep=values>=0;out[keep]=lookup[values[keep]]
    return out,[categories[int(i)] for i in present]

def csr_block(group, start, end):
    from scipy.sparse import csr_matrix
    p=group['indptr'][start:end+1].astype(np.int64)
    a,b=int(p[0]),int(p[-1])
    return csr_matrix((group['data'][a:b],group['indices'][a:b],p-p[0]),
                      shape=(end-start,int(group.attrs['shape'][1])))

def categorical(group):
    require(isinstance(group, __import__('h5py').Group),'Expected an AnnData categorical column')
    return group['codes'][:],group['categories'].asstr()[:].tolist()
