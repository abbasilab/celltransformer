"""Bundle audited runtime resources without changing the native source tree."""
import importlib.util
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / 'barseq_example'


def layout_module():
    spec = importlib.util.spec_from_file_location('ct2_build_layout', EXAMPLE / 'source_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildPy(build_py):
    def run(self):
        layout = layout_module()
        layout.audit_source()
        layout.audit_example()
        # Editable installs resolve resources in the checkout; wheels carry their own.
        if self.editable_mode:
            return
        else:
            super().run()
            bundle = Path(self.build_lib) / 'celltransformer2_barseq/_bundle'
            if bundle.exists():
                shutil.rmtree(bundle)
            bundle.mkdir(parents=True)
            import json
            manifest = json.loads((EXAMPLE / 'PACKAGE_MANIFEST.json').read_text())
            for name in [*manifest, 'PACKAGE_MANIFEST.json', 'LICENSE']:
                destination = bundle / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(EXAMPLE / name, destination)
            layout.copy_source(bundle)


setup(cmdclass={'build_py': BuildPy})
