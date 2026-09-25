#!/usr/bin/env python3
"""Refresh example hashes after reviewed adapter edits; never approve core changes."""
import argparse
from pathlib import Path
import hashlib
import importlib.util
import json

def update(name):
    example = Path(__file__).resolve().parents[1] / (name + '_example')
    spec = importlib.util.spec_from_file_location('ct2_manifest_' + name, example / 'source_layout.py')
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    layout.audit_source()
    paths = list(example.glob('*.py')) + list(example.glob('*.sh'))
    paths += [example / name for name in ['requirements.txt', 'CT2_SOURCE_MANIFEST.json', 'DOWNLOAD_MANIFEST.json']]
    paths += [example / name for name in ['README.md', 'VALIDATION.md'] if (example / name).is_file()]
    paths += [p for d in ['reference', 'environment'] for p in (example / d).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store']
    manifest = {str(p.relative_to(example)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (example / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{name}: updated {len(manifest)} adapter/dependency/reference file hashes; core source unchanged')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['all', 'allen', 'barseq'], default='all')
    args = parser.parse_args()
    for name in (['allen', 'barseq'] if args.example == 'all' else [args.example]):
        update(name)
