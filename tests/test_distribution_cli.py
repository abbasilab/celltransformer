"""Installed launcher routing without starting a GPU job or changing run state."""
import json
from pathlib import Path
import sys

import pytest

from celltransformer2_barseq import cli


@pytest.fixture
def invocation(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(cli,'version',lambda _: '0.1.0')
    monkeypatch.setattr(cli,'audit',lambda: {})
    monkeypatch.setattr(cli,'example_path',lambda: tmp_path/'bundle')
    monkeypatch.setattr(cli.platform,'system',lambda: 'Linux')
    monkeypatch.setattr(cli.platform,'machine',lambda: 'x86_64')
    monkeypatch.setattr(cli.shutil,'which',lambda _: '/bin/bash')
    monkeypatch.setattr(cli.os,'execvpe',lambda file,args,env: calls.append((file,args,env)))
    def run(*args):
        monkeypatch.setattr(sys,'argv',['celltransformer-barseq',*map(str,args)])
        cli.main()
        return calls[-1]
    return run


@pytest.mark.parametrize('background',[False,True])
def test_run_uses_requested_root_gpu_and_epochs(invocation,tmp_path,background):
    root=tmp_path/'experiment'
    args=['run','--root',root,'--gpu','2','--epochs','3']
    if background:args.append('--background')
    file,command,env=invocation(*args)
    assert file=='bash'
    assert env['BARSEQ_ROOT']==str(root.resolve())
    assert env['BARSEQ_GPU']=='2' and env['BARSEQ_EPOCHS']=='3'
    assert Path(command[1]).name==('launch_cactus.sh' if background else 'run_all.sh')
    if not background:assert command[2:]==['--root',str(root.resolve()),'--epochs','3']


@pytest.mark.parametrize('background',[False,True])
def test_resume_uses_saved_storage_root(invocation,tmp_path,background):
    run=tmp_path/'existing';run.mkdir()
    root=tmp_path/'original-storage'
    (run/'config.json').write_text(json.dumps({'root':str(root)}))
    args=['resume','--run',run,'--root',tmp_path/'different-storage','--gpu','1']
    if background:args.append('--background')
    _,command,env=invocation(*args)
    assert env['BARSEQ_ROOT']==str(root)
    assert env['BARSEQ_GPU']=='1'
    assert Path(command[1]).name==('resume_cactus.sh' if background else 'run_all.sh')
    assert command[2:]==([str(run)] if background else ['--resume',str(run)])


def test_reject_zero_epoch_run(invocation):
    with pytest.raises(SystemExit) as error:
        invocation('run','--epochs','0')
    assert error.value.code==2
