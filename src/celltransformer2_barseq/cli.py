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


EXAMPLES = {
    'allen': dict(directory='allen_example', bundle='_allen_bundle', prefix='ALLEN',
                  root='Allen1_CellTransformer2', title='Allen 1',
                  launch='launch.sh', resume='resume.sh', status='status.sh'),
    'barseq': dict(directory='barseq_example', bundle='_bundle', prefix='BARSEQ',
                   root='BARseq_CT2_example', title='BARseq',
                   launch='launch_cactus.sh', resume='resume_cactus.sh', status='status_cactus.sh'),
}


def example_path(example='barseq'):
    settings = EXAMPLES[example]
    package = Path(__file__).resolve().parent
    bundled = package / settings['bundle']
    if (bundled / 'PACKAGE_MANIFEST.json').is_file():
        return bundled
    checkout = package.parents[1] / settings['directory']
    if (checkout / 'PACKAGE_MANIFEST.json').is_file():
        return checkout
    raise RuntimeError(settings['title'] + ' runtime resources are missing; reinstall the package.')


def load_layout(example='barseq'):
    spec = importlib.util.spec_from_file_location('ct2_installed_layout_' + example, example_path(example) / 'source_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(example='barseq'):
    layout = load_layout(example)
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
                original_files=len(original),example=str(example_path(example)),native_package=str(native))


def train_main():
    """Resolve Hydra resources in a wheel; retain the native training implementation."""
    if not any(arg == '--config-path' or arg.startswith('--config-path=') for arg in sys.argv[1:]):
        layout = load_layout('allen')
        sys.argv[1:1] = ['--config-path', str(layout.source_root() / 'config')]
    from celltransformer.train.main import main as native_main
    native_main()


def main(example='barseq'):
    settings = EXAMPLES[example]
    prefix = settings['prefix']
    parser = argparse.ArgumentParser(description='Run the audited ' + settings['title'] + ' CellTransformer2 example.')
    parser.add_argument('--version', action='version', version=version('celltransformer2-test'))
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('audit',help='Verify installed source and bundled resource hashes.')
    for name in ['run','resume','status']:
        sub = commands.add_parser(name)
        sub.add_argument('--root',type=Path,default=Path(os.environ.get(prefix + '_ROOT',Path.home() / settings['root'])))
        if name != 'status':
            sub.add_argument('--gpu',default=os.environ.get(prefix + '_GPU','0'))
            sub.add_argument('--background',action='store_true',help='Detach with the existing nohup helper.')
        if name == 'run':
            sub.add_argument('--epochs',type=int,default=int(os.environ.get(prefix + '_EPOCHS','10')))
            if example == 'allen':
                sub.add_argument('--python',type=Path,help='Reuse an existing CT2 training environment without modifying it.')
                sub.add_argument('--rapids-python',type=Path,help='Reuse an existing RAPIDS environment without modifying it.')
                sub.add_argument('--prepare-only',action='store_true',help='Stop after preparing the data; resume this run to train.')
        if name == 'resume':
            sub.add_argument('--run',type=Path,help='Existing run directory; defaults to the latest run under --root.')
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(audit(example),indent=2))
        return
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('The complete ' + settings['title'] + ' workflow requires a Linux x86_64 GPU workstation.')
    if shutil.which('bash') is None:
        parser.error('Bash is required to run the workflow.')
    if args.command == 'run' and args.epochs <= 0:
        parser.error('--epochs must be positive')
    audit(example)
    base = example_path(example)
    env = os.environ.copy()
    env[prefix + '_ROOT'] = str(args.root.expanduser().resolve())
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH','')
    if args.command == 'status':
        command = ['bash',str(base / settings['status'])]
    else:
        env[prefix + '_GPU'] = args.gpu
        if args.command == 'run':
            env[prefix + '_EPOCHS'] = str(args.epochs)
            command = (['bash',str(base / settings['launch'])] if args.background else
                       ['bash',str(base / 'run_all.sh'),'--root',env[prefix + '_ROOT'],'--epochs',str(args.epochs)])
            if example == 'allen':
                for flag in ['python', 'rapids_python']:
                    path = getattr(args, flag)
                    if path is not None:
                        command.extend(['--' + flag.replace('_', '-'), str(path.expanduser().resolve())])
                if args.prepare_only:
                    command.append('--prepare-only')
        else:
            run = (args.run or Path((Path(env[prefix + '_ROOT']) / ('LATEST_' + prefix + '_CT2')).read_text().strip())).expanduser().resolve()
            saved = json.loads((run / 'config.json').read_text())
            expected = 'allen1_ct2_v1' if example == 'allen' else 'barseq_ct2_v1'
            if saved.get('workflow') != expected:
                parser.error('This run belongs to a different workflow; use its matching example command.')
            env[prefix + '_ROOT'] = saved['root']
            command = (['bash',str(base / settings['resume']),str(run)] if args.background else
                       ['bash',str(base / 'run_all.sh'),'--resume',str(run)])
    os.execvpe(command[0],command,env)


def allen_main():
    main('allen')
