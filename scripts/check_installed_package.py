"""Check a wheel outside its checkout, including bundled CPU report generation."""
import argparse
from importlib.metadata import distribution
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(output):
    import celltransformer
    from celltransformer2_barseq.cli import audit,example_path
    root=Path(sys.prefix).resolve()
    package=Path(celltransformer.__file__).resolve().parent
    assert package.is_relative_to(root),'Test must import the installed wheel, not the checkout'
    bundle=example_path()
    assert bundle.name=='_bundle' and bundle.is_relative_to(root)
    assert audit()['original_files']==66
    assert (bundle/'celltransformer2/config/train_redone_registration.yaml').is_file()
    assert (bundle/'reference/allen_structure_graph.json').is_file()
    assert (bundle/'environment/ct2-pip-freeze.txt').is_file()
    dist=distribution('celltransformer2-test')
    assert any(v.replace(' ','')=='torch==2.5.1' for v in dist.requires)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(prefix='ct2-installed-check-') as tmp:
        env['MPLCONFIGDIR']=tmp
        for entry in dist.entry_points:
            if entry.group!='console_scripts':continue
            command=Path(sys.executable).parent/entry.name
            completed=subprocess.run([str(command),'--help'],cwd=tmp,env=env,text=True,capture_output=True)
            assert completed.returncode==0,(entry.name,completed.stdout,completed.stderr)
            print(f'Installed command help: {entry.name}: PASSED',flush=True)
        subprocess.run([sys.executable,str(bundle/'test_adapter.py'),'--output',str(output)],cwd=tmp,env=env,check=True)
    assert (output/'report/review.html').is_file()
    assert (output/'report/results.zip').is_file()
    print('INSTALLED WHEEL: SOURCE, HYDRA CONFIG, COMMANDS, CPU ADAPTER AND REPORT: PASSED')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
