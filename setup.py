"""Bundle audited runtime resources without changing the native source tree."""
import importlib.util
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
EXAMPLES = {'allen_example': '_allen_bundle', 'barseq_example': '_bundle'}


def layout_module(example):
    spec = importlib.util.spec_from_file_location('ct2_build_layout_' + example.name, example / 'source_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildPy(build_py):
    def run(self):
        layouts = {name: layout_module(ROOT / name) for name in EXAMPLES}
        for layout in layouts.values():
            layout.audit_source()
            layout.audit_example()
        # Editable installs resolve resources in the checkout; wheels carry their own.
        if self.editable_mode:
            return
        else:
            super().run()
            import json
            for example_name, bundle_name in EXAMPLES.items():
                example = ROOT / example_name
                bundle = Path(self.build_lib) / 'celltransformer2_barseq' / bundle_name
                if bundle.exists():
                    shutil.rmtree(bundle)
                bundle.mkdir(parents=True)
                manifest = json.loads((example / 'PACKAGE_MANIFEST.json').read_text())
                for name in [*manifest, 'PACKAGE_MANIFEST.json', 'LICENSE']:
                    destination = bundle / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(example / name, destination)
                layouts[example_name].copy_source(bundle)


setup(cmdclass={'build_py': BuildPy})
