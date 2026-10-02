"""Run with PYTHONPATH=src python -m unittest discover -s tests -p test_notebook_report.py.

The end-to-end tests execute only the newly added cells, against synthetic
completed native outputs. No model training or GPU is involved.
"""
import hashlib
import json
import os
from pathlib import Path
from contextlib import chdir
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from celltransformer.notebooks import helper
from celltransformer.notebooks import reporting as notebook_report
from celltransformer.notebooks._report import pearson_rows

HERE = Path(__file__).resolve().parent
NOTEBOOKS = Path(os.environ.get('CT2_TEST_NOTEBOOKS', HERE.parent/'notebooks'))


def make_history(checkpoint, epochs):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    info = checkpoint.parent / 'wandb_run.json'
    info.write_text(json.dumps({'run_id': 'fixture'}))
    path = checkpoint.parent / 'wandb/offline-run-fixture/run-fixture.wandb'
    path.parent.mkdir(parents=True)
    store = DataStore()
    store.open_for_write(str(path))
    for epoch in range(epochs):
        for values in [dict(epoch=epoch, **{'train/nll_step': 999}),
                       dict(epoch=epoch, **{'train/nll_epoch': .8/(epoch+1)}),
                       dict(epoch=epoch, **{'valid/nll_epoch': .9/(epoch+1)})]:
            record = wandb_internal_pb2.Record()
            for key, value in values.items():
                item = record.history.item.add()
                item.key = key
                item.value_json = json.dumps(value)
            store.write(record)
    store.close()
    return path


def native_fixture(root, notebook):
    full = notebook.startswith('01')
    ks = [25, 354, 670] if full else [20, 50]
    tags = ['allen_full', 'allen_full_smoothed'] if full else ['custom', 'custom_smoothed']
    work = root / 'work'
    work.mkdir()
    embeddings = {tags[0]: work/'sections', tags[1]: work/'sections_smoothed'}
    for directory in embeddings.values():
        directory.mkdir()
    domains = work/'domains'
    domains.mkdir()
    rng = np.random.default_rng(5)
    n = 120
    ids = [f'cell-{i}' for i in range(n*2)]
    meta = pd.DataFrame({'cell_id': ids, 'section': ['sample.1']*n + ['sample.2']*n,
                         'x': rng.uniform(0, 300, n*2), 'y': rng.uniform(0, 300, n*2),
                         'cell_type': [f'type-{i%4}' for i in range(n*2)]})
    mask = np.ones(n*2, bool)
    mask[0] = False
    for tag, directory in embeddings.items():
        for i in range(2):
            stem = f'section_{i}'
            frame = meta.iloc[i*n:(i+1)*n]
            frame.to_parquet(directory/(stem+'.parquet'), index=False)
            np.savez(directory/(stem+'_embeddings.npz'), embeddings=np.ones((n,4), np.float32), has_neighbors=mask[i*n:(i+1)*n])
        for k in ks:
            np.save(domains/f'{tag}_kmeans_k{k}_labels.npy', np.arange(n*2,dtype=np.int32)%k)
    reference = pd.DataFrame({'cell_label': ids, 'parcellation_division': ['Isocortex']*(n*2),
                              'parcellation_structure': ['MO' if i%2 else 'SS' for i in range(n*2)],
                              'parcellation_substructure': ['NA' if i%3==0 else f'layer-{i%3}' for i in range(n*2)]})
    raw = work/'reference.csv'
    reference.sample(frac=1, random_state=22).to_csv(raw, index=False)
    checkpoint = work/'checkpoints/finished/last.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('Synthetic placeholder, not a trained checkpoint')
    epochs = 10 if full else 1
    make_history(checkpoint, epochs)
    namespace = dict(Path=Path, WORK_DIR=work, EMBEDDING_SETS=embeddings, DOMAINS_DIR=domains,
                     SMOOTHED_DIR=embeddings[tags[1]], RAW_METADATA=raw, CELL_ID_COL='cell_label',
                     CHECKPOINT=checkpoint, EPOCHS=epochs, VALID_SECTIONS=['sample.2'], KS=ks,
                     N_GENES=4, PATCH_SIZE=85., SIGMA=12.01, N_NEIGHBORS=500)
    return namespace


class NotebookReportTests(unittest.TestCase):
    def test_original_cells_and_metadata_unchanged(self):
        proof = json.loads((HERE/'fixtures/notebook_report_preservation.json').read_text())
        for name, expected in proof['notebooks'].items():
            doc = json.loads((NOTEBOOKS/name).read_text())
            cells = doc.pop('cells')
            reverse = {new: old for old, new in expected['allowed_source_replacements']}
            restored = []
            for cell in cells:
                original = dict(cell)
                if isinstance(cell['source'], list):
                    original['source'] = [reverse.get(line, line) for line in cell['source']]
                restored.append(hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest())
            self.assertEqual(restored, expected['original_cell_sha256'])
            self.assertEqual(hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest(), expected['notebook_metadata_sha256'])
            new = [cell for cell in cells if cell.get('id', '').startswith('step13-')]
            self.assertEqual(len(new), 3)
            for cell in new:
                if cell['cell_type'] == 'code':
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])

    def test_helper_code_is_unchanged(self):
        proof = json.loads((HERE/'fixtures/notebook_report_preservation.json').read_text())
        self.assertEqual(hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(), proof['helper_sha256'])

    def test_notebook_folder_has_no_companion_files(self):
        files = [p for p in NOTEBOOKS.iterdir() if p.is_file()]
        self.assertTrue({'01_allen_dataset.ipynb', '02_custom_dataset.ipynb'} <=
                        {p.name for p in files})
        self.assertEqual(sorted(p.name for p in files if p.suffix != '.ipynb'), [])

    def test_helper_works_away_from_the_notebooks(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            self.assertEqual(helper.safe_section('example.1'), 'example_1')
            metadata = Path(directory)/'section.parquet'
            pd.DataFrame({'x': [0., 1., 10.], 'y': [0., 1., 10.]}).to_parquet(metadata)
            np.testing.assert_array_equal(np.sort(helper.neighbor_counts(metadata, 4)), [0, 1, 1])

    def run_notebook_step13(self, filename, with_reference=True):
        with tempfile.TemporaryDirectory(prefix='ct2-step13-') as directory:
            ns = native_fixture(Path(directory), filename)
            paths = [p for p in ns['WORK_DIR'].rglob('*') if p.is_file()]
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            notebook = json.loads((NOTEBOOKS/filename).read_text())
            cells = {cell.get('id'): ''.join(cell.get('source', [])) for cell in notebook['cells']}
            actual_generate = notebook_report.generate_report
            def synthetic_report(**kwargs):
                kwargs['synthetic'] = True
                return actual_generate(**kwargs)
            with chdir(directory), patch.object(notebook_report, 'generate_report', side_effect=synthetic_report), patch.object(notebook_report, 'show_report'):
                exec(compile(cells['step13-settings'], filename+':Step13 settings', 'exec'), ns)
                if not with_reference:
                    ns['REFERENCE_COLUMNS'] = {}
                exec(compile(cells['step13-generate'], filename+':Step13 generate', 'exec'), ns)
            html = ns['report_html'].read_text()
            self.assertIn('SYNTHETIC SOFTWARE TEST', html)
            self.assertIn('<h2>Training diagnostics</h2>', html)
            self.assertIn('<h2>Spatial domains</h2>', html)
            self.assertEqual('<h2>Reference–domain correspondence</h2>' in html, with_reference)
            self.assertNotIn('<h2>Composition matrices', html)
            summary = json.loads(ns['report_html'].with_name('summary.json').read_text())
            self.assertEqual(summary['correspondence_panels'], len(ns['KS'])*2 if with_reference else 0)
            history = pd.read_csv(ns['report_html'].parent/'tables/training_history.tsv', sep='\t')
            np.testing.assert_allclose(history.training_nll, [.8/(i+1) for i in range(ns['EPOCHS'])])
            np.testing.assert_allclose(history.validation_nll, [.9/(i+1) for i in range(ns['EPOCHS'])])
            if with_reference:
                palette = pd.read_csv(ns['report_html'].parent/'reference_colors.tsv', sep='\t', keep_default_na=False)
                self.assertIn('NA', palette.label.tolist())
            for path, sha in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)

    def test_allen_added_cells(self):
        self.run_notebook_step13('01_allen_dataset.ipynb')

    def test_custom_added_cells(self):
        self.run_notebook_step13('02_custom_dataset.ipynb')

    def test_custom_without_reference(self):
        self.run_notebook_step13('02_custom_dataset.ipynb', with_reference=False)

    def test_missing_history_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            history = notebook_report.read_training_history(Path(directory)/'last.ckpt', expected_epochs=2)
            self.assertTrue(history['notes'])
            self.assertTrue(all(row['training_nll'] is None and row['validation_nll'] is None for row in history['history']))

    def test_reference_join_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = native_fixture(Path(directory), '02_custom_dataset.ipynb')
            refs = pd.read_csv(ns['RAW_METADATA'], keep_default_na=False)
            refs = pd.concat([refs, refs.iloc[:1]])
            with self.assertRaisesRegex(ValueError, 'unique'):
                notebook_report.report_from_notebook(work_dir=ns['WORK_DIR'], embedding_sets=ns['EMBEDDING_SETS'],
                    domains_dir=ns['DOMAINS_DIR'], ks=ns['KS'], reference_metadata=refs,
                    reference_id_column='cell_label', reference_columns={'Reference':'parcellation_substructure'})

    def test_pearson_and_undefined_compositions(self):
        corr = pearson_rows(np.eye(3), np.eye(3))
        np.testing.assert_allclose(np.diag(corr), 1)
        self.assertAlmostEqual(corr[0,1], -.5)
        self.assertTrue(np.isnan(pearson_rows(np.zeros((1,3)), np.eye(3))).all())


if __name__ == '__main__':
    unittest.main()
