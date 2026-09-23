#!/usr/bin/env python3
"""Refresh example hashes after reviewed adapter edits; never approve core changes."""
from pathlib import Path
import hashlib
import json
import sys

example = Path(__file__).resolve().parents[1] / 'barseq_example'
sys.path.insert(0, str(example))
from source_layout import audit_source

audit_source()
paths = list(example.glob('*.py')) + list(example.glob('*.sh'))
paths += [example / name for name in ['requirements.txt', 'CT2_SOURCE_MANIFEST.json', 'DOWNLOAD_MANIFEST.json']]
paths += [p for d in ['reference', 'environment'] for p in (example / d).rglob('*') if p.is_file()]
manifest = {str(p.relative_to(example)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(example / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Updated {len(manifest)} adapter/dependency/reference file hashes; core source unchanged')
