"""Check a wheel outside its checkout, including bundled CPU report generation."""
import argparse
from importlib.metadata import distribution
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main(output):
    import celltransformer
    from celltransformer.notebooks import helper, reporting
    from celltransformer.notebooks import _report
    from celltransformer2_barseq.cli import audit,example_path
    root=Path(sys.prefix).resolve()
    package=Path(celltransformer.__file__).resolve().parent
    assert package.is_relative_to(root),'Test must import the installed wheel, not the checkout'
    for module in [helper, reporting, _report]:
        assert Path(module.__file__).resolve().is_relative_to(package)
    bundles={example:example_path(example) for example in ['allen','barseq']}
    for example,bundle in bundles.items():
        assert bundle.name==('_allen_bundle' if example=='allen' else '_bundle') and bundle.is_relative_to(root)
        assert audit(example)['original_files']==66
        assert (bundle/'celltransformer2/config/train_redone_registration.yaml').is_file()
        assert (bundle/'environment/ct2-pip-freeze.txt').is_file()
    assert (bundles['barseq']/'reference/allen_structure_graph.json').is_file()
    assert (bundles['allen']/'DOWNLOAD_MANIFEST.json').is_file()
    dist=distribution('celltransformer2-test')
    assert any(v.replace(' ','')=='torch==2.5.1' for v in dist.requires)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(prefix='ct2-installed-check-') as tmp:
        env['MPLCONFIGDIR']=tmp
        # Exercise the actual notebook cells with only the .ipynb files in
        # their folder; helpers must resolve from the installed wheel.
        source=Path(__file__).resolve().parents[1]
        standalone=Path(tmp)/'standalone'
        (standalone/'notebooks').mkdir(parents=True)
        (standalone/'tests/fixtures').mkdir(parents=True)
        for notebook in sorted((source/'notebooks').glob('*.ipynb')):
            shutil.copy2(notebook, standalone/'notebooks'/notebook.name)
        shutil.copy2(source/'tests/test_notebook_report.py', standalone/'tests/test_notebook_report.py')
        shutil.copy2(source/'tests/fixtures/notebook_report_preservation.json', standalone/'tests/fixtures/notebook_report_preservation.json')
        env.pop('CT2_TEST_NOTEBOOKS',None)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_notebook_report.py','-v'],
                       cwd=standalone,env=env,check=True)
        assert len([entry for entry in dist.entry_points if entry.group=='console_scripts'])==8
        for entry in dist.entry_points:
            if entry.group!='console_scripts':continue
            command=Path(sys.executable).parent/entry.name
            completed=subprocess.run([str(command),'--help'],cwd=tmp,env=env,text=True,capture_output=True)
            assert completed.returncode==0,(entry.name,completed.stdout,completed.stderr)
            print(f'Installed command help: {entry.name}: PASSED',flush=True)
        for example,bundle in bundles.items():
            destination=output/example
            subprocess.run([sys.executable,str(bundle/'test_adapter.py'),'--output',str(destination)],cwd=tmp,env=env,check=True)
            assert (destination/'report/review.html').is_file()
            assert (destination/'report/results.zip').is_file()
    print('INSTALLED WHEEL: STANDALONE NOTEBOOK REPORTS, BOTH EXAMPLES, SOURCE, HYDRA CONFIG, COMMANDS AND CPU ADAPTERS: PASSED')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
