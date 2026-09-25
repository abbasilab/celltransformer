"""Installed launcher routing without starting a GPU job or changing run state."""
import json
from pathlib import Path
import sys

import pytest

from celltransformer2_barseq import cli


@pytest.fixture(params=['allen','barseq'])
def invocation(request,monkeypatch,tmp_path):
    example=request.param
    calls=[]
    monkeypatch.setattr(cli,'version',lambda _: '0.2.0')
    monkeypatch.setattr(cli,'audit',lambda which: {} if which==example else pytest.fail('Wrong example audited'))
    monkeypatch.setattr(cli,'example_path',lambda which: tmp_path/which)
    monkeypatch.setattr(cli.platform,'system',lambda: 'Linux')
    monkeypatch.setattr(cli.platform,'machine',lambda: 'x86_64')
    monkeypatch.setattr(cli.shutil,'which',lambda _: '/bin/bash')
    monkeypatch.setattr(cli.os,'execvpe',lambda file,args,env: calls.append((file,args,env)))
    def run(*args):
        monkeypatch.setattr(sys,'argv',['celltransformer-'+example,*map(str,args)])
        cli.main(example)
        return calls[-1]
    run.example=example
    return run


@pytest.mark.parametrize('background',[False,True])
def test_run_uses_requested_root_gpu_and_epochs(invocation,tmp_path,background):
    root=tmp_path/'experiment'
    args=['run','--root',root,'--gpu','2','--epochs','3']
    if background:args.append('--background')
    file,command,env=invocation(*args)
    settings=cli.EXAMPLES[invocation.example]
    prefix=settings['prefix']
    assert file=='bash'
    assert env[prefix+'_ROOT']==str(root.resolve())
    assert env[prefix+'_GPU']=='2' and env[prefix+'_EPOCHS']=='3'
    assert Path(command[1]).name==(settings['launch'] if background else 'run_all.sh')
    if not background:assert command[2:]==['--root',str(root.resolve()),'--epochs','3']


@pytest.mark.parametrize('background',[False,True])
def test_resume_uses_saved_storage_root(invocation,tmp_path,background):
    run=tmp_path/'existing';run.mkdir()
    root=tmp_path/'original-storage'
    workflow='allen1_ct2_v1' if invocation.example=='allen' else 'barseq_ct2_v1'
    (run/'config.json').write_text(json.dumps({'root':str(root),'workflow':workflow}))
    args=['resume','--run',run,'--root',tmp_path/'different-storage','--gpu','1']
    if background:args.append('--background')
    _,command,env=invocation(*args)
    settings=cli.EXAMPLES[invocation.example]
    assert env[settings['prefix']+'_ROOT']==str(root)
    assert env[settings['prefix']+'_GPU']=='1'
    assert Path(command[1]).name==(settings['resume'] if background else 'run_all.sh')
    assert command[2:]==([str(run)] if background else ['--resume',str(run)])


def test_reject_zero_epoch_run(invocation):
    with pytest.raises(SystemExit) as error:
        invocation('run','--epochs','0')
    assert error.value.code==2


def test_reject_resume_from_other_workflow(invocation,tmp_path):
    run=tmp_path/'wrong-run';run.mkdir()
    workflow='barseq_ct2_v1' if invocation.example=='allen' else 'allen1_ct2_v1'
    (run/'config.json').write_text(json.dumps({'root':str(tmp_path),'workflow':workflow}))
    with pytest.raises(SystemExit) as error:
        invocation('resume','--run',run)
    assert error.value.code==2


def test_status_uses_requested_root(invocation,tmp_path):
    _,command,env=invocation('status','--root',tmp_path)
    settings=cli.EXAMPLES[invocation.example]
    assert env[settings['prefix']+'_ROOT']==str(tmp_path.resolve())
    assert Path(command[1]).name==settings['status']


@pytest.mark.parametrize('background',[False,True])
def test_allen_environment_reuse_and_prepare_only(invocation,tmp_path,background):
    if invocation.example!='allen':
        pytest.skip('These environment-reuse options are specific to Allen.')
    args=['run','--python',tmp_path/'ct2/bin/python','--rapids-python',tmp_path/'rapids/bin/python','--prepare-only']
    if background:args.append('--background')
    _,command,_=invocation(*args)
    assert command[-5:]==['--python',str(tmp_path/'ct2/bin/python'),'--rapids-python',str(tmp_path/'rapids/bin/python'),'--prepare-only']
