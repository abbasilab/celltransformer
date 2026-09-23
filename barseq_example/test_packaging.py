"""Integrity and presentation contracts; no GPU or downloaded matrix required."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from source_layout import HERE, audit_source, copy_source, digest
from reference_colors import distinct_references


def test_original_source_and_snapshot(tmp_path):
    manifest = audit_source()
    assert len(manifest) == 66
    copy_source(tmp_path)
    for name, expected in manifest.items():
        assert digest(tmp_path / 'celltransformer2' / name) == expected


def test_palette_is_unique_and_keeps_anatomical_identity():
    refs = json.loads((HERE / 'results/references.json').read_text())
    graph = json.loads((HERE / 'reference/allen_structure_graph.json').read_text())
    colored = distinct_references(refs, graph)
    combined = {}
    for level, info in colored.items():
        assert len(set(info['colors'])) == info['k']
        for key in refs[level]:
            if key != 'colors':
                assert info[key] == refs[level][key]
        for structure, color in zip(info['ids'], info['colors']):
            assert combined.setdefault(structure, color) == color
    assert len(set(combined.values())) == len(combined)
    fine = dict(zip(colored['released']['names'], colored['released']['colors']))
    motor = [fine[n] for n in ['MOp1', 'MOp2/3', 'MOp5', 'MOp6a', 'MOp6b']]
    assert len(set(motor)) == 5


def test_cublas_is_set_before_python_starts(tmp_path):
    shutil.copy2(HERE / 'run_all.sh', tmp_path / 'run_all.sh')
    (tmp_path / 'pipeline.py').write_text(
        'import os, json\nprint(json.dumps(dict(cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),'
        'gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))))\n')
    env = os.environ.copy()
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env['BARSEQ_GPU'] = '1'
    result = subprocess.run(['bash', str(tmp_path / 'run_all.sh')], env=env,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'cublas': ':4096:8', 'gpu': '1'}
