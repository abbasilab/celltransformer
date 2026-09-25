"""Resolve the single repository source tree or an immutable run snapshot."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
RELOCATED = {'README.md', '.gitignore', '.roborev.toml', 'pyproject.toml'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_root(base=HERE):
    embedded = Path(base) / 'celltransformer2'
    return embedded if (embedded / 'src').is_dir() else Path(base).parent


def original_file(name, base=HERE):
    root = source_root(base)
    if root == Path(base).parent and name in RELOCATED:
        return root / 'provenance/upstream' / name
    return root / name


def audit_source(base=HERE):
    manifest = json.loads((Path(base) / 'CT2_SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest.items():
        path = original_file(name, base)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('CellTransformer2 source differs from demonstrated version: ' + name)
    return manifest


def copy_source(destination, base=HERE):
    manifest = audit_source(base)
    for name in manifest:
        target = Path(destination) / 'celltransformer2' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_file(name, base), target)


def audit_example(base=HERE):
    manifest = json.loads((Path(base) / 'PACKAGE_MANIFEST.json').read_text())
    for name, expected in manifest.items():
        if digest(Path(base) / name) != expected:
            raise ValueError('Changed Allen example file: ' + name)


if __name__ == '__main__':
    audit_example()
    manifest = audit_source()
    print(f'SOURCE AND EXAMPLE AUDIT: PASSED ({len(manifest)} original source files)')
