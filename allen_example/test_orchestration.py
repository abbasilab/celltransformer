"""No-GPU coordinator test: mocked stages, real snapshots/status/recovery."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import pipeline


def test_failed_training_resumes_without_repreparing(monkeypatch,tmp_path):
    root=tmp_path/'run-root';called=[];fail_training=[True]
    class Child:
        def __init__(self,cmd,**kwargs):
            self.pid=12345;self.code=0
            name=Path(cmd[2]).stem if len(cmd)>2 and cmd[1]=='-u' else 'environment'
            called.append(name)
            run=Path((root/'LATEST_ALLEN_CT2').read_text().strip())
            if name=='train_ct2' and fail_training[0]:
                fail_training[0]=False;self.code=1;return
            outputs={'prepare_allen':['prepared/summary.json'],
                'train_ct2':['celltransformer/summary.json','celltransformer/embeddings.npy',
                            'celltransformer/checkpoints/last.ckpt','celltransformer/epochs.json'],
                'cluster':['analysis/labels/test.npy','analysis/clustering.json'],
                'report':['report/review.html','report/results.zip']}
            if name=='analyze':
                paths=['analysis/smoothed.npy'] if cmd[-1]=='smooth' else ['analysis/metrics/test.tsv']
            else:paths=outputs.get(name,[])
            for relative in paths:
                p=run/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('TEST STUB; NOT DATA')
        def wait(self):return self.code
        def poll(self):return self.code
    monkeypatch.setattr(pipeline.subprocess,'Popen',Child)
    # Stages write tiny stubs; model available storage independently of CI disk size.
    monkeypatch.setattr(pipeline.shutil,'disk_usage',lambda root:SimpleNamespace(free=100*1024**3))
    # Isolate locking from exception traceback lifetimes in the test process.
    monkeypatch.setattr(pipeline,'lock_root',lambda root:None)
    monkeypatch.setattr(pipeline.fcntl,'flock',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['pipeline.py','--root',str(root),'--minimum-free-gib','0'])
    with pytest.raises(RuntimeError,match='train failed'):pipeline.main()
    run=Path((root/'LATEST_ALLEN_CT2').read_text().strip())
    status=json.loads((run/'status.json').read_text())
    assert status['overall']=='FAILED'
    assert status['stages']['prepare']['status']=='PASSED'
    cfg=json.loads((run/'config.json').read_text())
    assert cfg['epochs']==10 and cfg['model']['expression_dim']==500
    assert cfg['storage_reserve_gib']==20
    assert cfg['patch_width_um']==85 and cfg['validation_fraction']==.2
    monkeypatch.setattr(sys,'argv',['pipeline.py','--resume',str(run)])
    pipeline.main()
    status=json.loads((run/'status.json').read_text())
    assert status['overall']=='PASSED' and 'error' not in status
    assert called.count('prepare_allen')==1
    assert called.count('train_ct2')==2
    assert (run/'SUCCESS').read_text().strip()==str(run/'report/review.html')
    assert (run/'runner_snapshot/celltransformer2/src/celltransformer/train/main.py').is_file()
