"""Thin installed-package entry points; scientific stages remain in the snapshot."""
import argparse
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def example_path():
    package = Path(__file__).resolve().parent
    bundled = package / '_bundle'
    if (bundled / 'PACKAGE_MANIFEST.json').is_file():
        return bundled
    checkout = package.parents[1] / 'barseq_example'
    if (checkout / 'PACKAGE_MANIFEST.json').is_file():
        return checkout
    raise RuntimeError('BARseq runtime resources are missing; reinstall the package.')


def load_layout():
    spec = importlib.util.spec_from_file_location('ct2_installed_layout', example_path() / 'source_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit():
    layout = load_layout()
    layout.audit_example()
    original = layout.audit_source()
    # Check both the runtime snapshot source and the normally imported model package.
    native = Path(importlib.util.find_spec('celltransformer').origin).parent
    for name, expected in original.items():
        prefix = 'src/celltransformer/'
        if name.startswith(prefix):
            if layout.digest(native / name[len(prefix):]) != expected:
                raise ValueError('Installed native source differs: ' + name)
    return dict(status='PASSED',version=version('celltransformer2-test'),
                original_files=len(original),example=str(example_path()),native_package=str(native))


def train_main():
    """Resolve Hydra resources in a wheel; retain the native training implementation."""
    if not any(arg == '--config-path' or arg.startswith('--config-path=') for arg in sys.argv[1:]):
        layout = load_layout()
        sys.argv[1:1] = ['--config-path', str(layout.source_root() / 'config')]
    from celltransformer.train.main import main as native_main
    native_main()


def main():
    parser = argparse.ArgumentParser(description='Run the audited BARseq CellTransformer2 example.')
    parser.add_argument('--version', action='version', version=version('celltransformer2-test'))
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('audit',help='Verify installed source and bundled resource hashes.')
    for name in ['run','resume','status']:
        sub = commands.add_parser(name)
        sub.add_argument('--root',type=Path,default=Path(os.environ.get('BARSEQ_ROOT',Path.home() / 'BARseq_CT2_example')))
        if name != 'status':
            sub.add_argument('--gpu',default=os.environ.get('BARSEQ_GPU','0'))
            sub.add_argument('--background',action='store_true',help='Detach with the existing nohup helper.')
        if name == 'run':
            sub.add_argument('--epochs',type=int,default=int(os.environ.get('BARSEQ_EPOCHS','10')))
        if name == 'resume':
            sub.add_argument('--run',type=Path,help='Existing run directory; defaults to the latest run under --root.')
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(audit(),indent=2))
        return
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('The complete BARseq workflow requires a Linux x86_64 GPU workstation.')
    if shutil.which('bash') is None:
        parser.error('Bash is required to run the workflow.')
    if args.command == 'run' and args.epochs <= 0:
        parser.error('--epochs must be positive')
    audit()
    base = example_path()
    env = os.environ.copy()
    env['BARSEQ_ROOT'] = str(args.root.expanduser().resolve())
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH','')
    if args.command == 'status':
        command = ['bash',str(base / 'status_cactus.sh')]
    else:
        env['BARSEQ_GPU'] = args.gpu
        if args.command == 'run':
            env['BARSEQ_EPOCHS'] = str(args.epochs)
            command = (['bash',str(base / 'launch_cactus.sh')] if args.background else
                       ['bash',str(base / 'run_all.sh'),'--root',env['BARSEQ_ROOT'],'--epochs',str(args.epochs)])
        else:
            run = (args.run or Path((Path(env['BARSEQ_ROOT']) / 'LATEST_BARSEQ_CT2').read_text().strip())).expanduser().resolve()
            env['BARSEQ_ROOT'] = json.loads((run / 'config.json').read_text())['root']
            command = (['bash',str(base / 'resume_cactus.sh'),str(run)] if args.background else
                       ['bash',str(base / 'run_all.sh'),'--resume',str(run)])
    os.execvpe(command[0],command,env)
