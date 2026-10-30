"""Repository paths for the example folders, and the celltransformer-train entry point."""
from pathlib import Path
import sys


EXAMPLE_DIRS = {'allen': 'allen_example', 'barseq': 'barseq_example'}


def example_path(example='barseq'):
    '''The example folder (allen_example/ or barseq_example/) in the repository checkout.'''
    folder = Path(__file__).resolve().parents[2] / EXAMPLE_DIRS[example]
    if not folder.is_dir():
        raise RuntimeError(f'{folder} not found; install the package with pip install -e . from the repository.')
    return folder


def train_main():
    """Point Hydra at the repository's config/ unless --config-path is given; run the native training."""
    if not any(arg == '--config-path' or arg.startswith('--config-path=') for arg in sys.argv[1:]):
        config = Path(__file__).resolve().parents[2] / 'config'
        sys.argv[1:1] = ['--config-path', str(config)]
    from celltransformer.train.main import main as native_main
    native_main()
