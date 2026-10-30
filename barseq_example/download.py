"""Resumable, checksum-verified downloads of the BARseq pilot release, converted into notebook 02's raw input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fetch(spec, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest.with_name(dest.name + '.verified.json')
    if dest.exists():
        stat = dest.stat()
        if marker.exists():
            old = json.loads(marker.read_text())
            if old == dict(spec=spec, size=stat.st_size, mtime_ns=stat.st_mtime_ns):
                return
        if stat.st_size != spec['size'] or digest(dest) != spec['sha256']:
            raise ValueError(f'Existing file fails the pinned checksum: {dest}; it was not overwritten')
    else:
        part = dest.with_name(dest.name + '.partial')
        missing = spec['size'] - (part.stat().st_size if part.exists() else 0)
        if missing < 0:
            raise ValueError(f'Partial download is larger than the pinned file: {part}')
        if shutil.disk_usage(dest.parent).free < missing + 5 * 1024**3:
            raise ValueError(f'Insufficient space for {dest.name} plus a 5-GiB reserve')
        if missing:
            subprocess.run(['curl', '--fail', '--location', '--retry', '8', '--retry-delay', '5',
                            '--connect-timeout', '60', '--continue-at', '-', '--output', str(part), spec['url']], check=True)
        if part.stat().st_size != spec['size'] or digest(part) != spec['sha256']:
            raise ValueError(f'Download failed the published size/SHA-256 check: {part}')
        part.replace(dest)
    stat = dest.stat()
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(spec=spec, size=stat.st_size, mtime_ns=stat.st_mtime_ns), indent=2))
    tmp.replace(marker)


def structure_tree():
    '''Allen CCF structures by ID, each with its path of ancestor IDs from the root.'''
    nodes = {}

    def walk(node, ancestors):
        path = ancestors + [node['id']]
        nodes[node['id']] = {**{key: value for key, value in node.items() if key != 'children'}, 'path': path}
        for child in node.get('children', []):
            walk(child, path)

    graph = json.loads((HERE / 'reference/allen_structure_graph.json').read_text())
    for root in graph['msg']:
        walk(root, [])
    return nodes


def match_cells(matrix, labels):
    '''Join the published cell labels to the released count matrix by cell ID.'''
    matrix_ids = np.asarray(matrix['id'], np.int64)
    require(len(np.unique(matrix_ids)) == len(matrix_ids), 'Duplicate matrix cell IDs')

    #label sample IDs are "<section>_<cell ID>"
    parts = labels['sample'].str.extract(r'^(\d+)_(\d+)$')
    require(parts.notna().all().all(), 'Malformed label sample IDs')
    label_ids = parts[1].astype(np.int64)
    require(not label_ids.duplicated().any(), 'Duplicate label cell IDs')

    #keep matrix cells that have a published label
    label_rows = pd.Index(label_ids).get_indexer(matrix_ids)
    keep = label_rows >= 0
    require(np.array_equal(keep, np.asarray(matrix['clustid']) != 'qc-filtered'),
            'Published labels disagree with the released QC assignments')
    cells = labels.iloc[label_rows[keep]].reset_index(drop=True)
    require(np.array_equal(parts[0].iloc[label_rows[keep]].astype(int), matrix['slice'][keep]),
            'Cell-ID join crosses sections')
    n_unmatched = int((~label_ids.isin(matrix_ids)).sum())
    return keep, cells, n_unmatched


def cell_types(cells):
    '''Published class, subclass and cluster; a missing subclass or cluster falls back to the class.'''
    cells = cells.rename(columns={'class': 'cell_class', 'subclass': 'cell_subclass', 'cluster': 'cell_cluster'}).copy()
    require(not cells.cell_class.isin(['NA', '']).any(), 'Missing class cannot supply a fallback type')
    for column in ['cell_subclass', 'cell_cluster']:
        absent = cells[column].isin(['NA', ''])
        cells[column + '_fallback_to_class'] = absent
        cells.loc[absent, column] = cells.loc[absent, 'cell_class']
    cells['qc_flag'] = cells.cell_subclass.eq('Low Quality')
    return cells


def sequencing_genes(matrix):
    '''The 106 named sequencing targets, without the unused barcodes and hybridization channels.'''
    genes = np.asarray(matrix['genes'])[:, 0].tolist()
    require(len(set(genes)) == len(genes), 'Duplicate matrix gene names')
    chosen = json.loads((HERE / 'reference/sequencing_genes.json').read_text())
    require(not any(gene.startswith('unused') or gene in ['Slc17a7', 'Gad1', 'Slc30a3'] for gene in chosen),
            'Control or hybridization channel selected')
    columns = pd.Index(genes).get_indexer(chosen)
    require((columns >= 0).all(), 'A pinned sequencing gene is absent')
    return chosen, columns


def convert(raw, out):
    '''Write the raw counts (.h5ad) and per-cell metadata (.parquet) that notebook 02 reads.'''
    #load counts and published labels
    matrix = loadmat(raw / 'wholecortexBARseq_filt_neurons_fixedbent_CCF.mat', simplify_cells=True)['filt_neurons']
    labels = pd.read_csv(raw / 'labels_20211208_withgaba_nonexc.csv', keep_default_na=False)
    keep, cells, n_unmatched = match_cells(matrix, labels)
    cells = cell_types(cells)

    #cell IDs as strings (they become the .h5ad obs_names), sections and µm coordinates
    cells['cell_id'] = np.asarray(matrix['id'][keep], np.int64).astype(str)
    section_numbers = np.asarray(matrix['slice'][keep], int)
    cells['section_number'] = section_numbers
    cells['section'] = [f'Slice{number:02d}' for number in section_numbers]
    xy = np.asarray(matrix['fixedpos'][keep], np.float64)
    require(np.isfinite(xy).all(), 'Nonfinite cell coordinates')
    cells['x_um'] = xy[:, 0]
    cells['y_um'] = xy[:, 1]

    #raw integer counts for the sequencing genes
    genes, columns = sequencing_genes(matrix)
    counts = sparse.csr_matrix(matrix['expmat'][keep][:, columns])
    counts.eliminate_zeros()
    require(np.isfinite(counts.data).all() and (counts.data >= 0).all(), 'Counts must be finite and nonnegative')
    require(np.equal(counts.data, np.floor(counts.data)).all(), 'Counts must be integers')
    require(counts.data.max() <= np.iinfo(np.int32).max, 'Count integer overflow')
    counts = counts.astype(np.int32)
    cells['total_counts'] = np.asarray(counts.sum(1)).ravel().astype(np.int64)
    cells['n_genes'] = np.diff(counts.indptr)

    #released region names, joined by cell ID and checked against the matrix
    registration = pd.read_csv(raw / 'neurons_registered.csv', keep_default_na=False,
                               usecols=['id', 'slice', 'pos_x', 'pos_y', 'name', 'x_CCF', 'y_CCF', 'z_CCF'])
    require(not registration.id.duplicated().any(), 'Duplicate registration IDs')
    registration_rows = pd.Index(registration.id.astype(str)).get_indexer(cells.cell_id)
    require((registration_rows >= 0).all(), 'A cell lacks a registration record')
    registration = registration.iloc[registration_rows].reset_index(drop=True)
    require(np.array_equal(registration['slice'], section_numbers), 'Registration IDs cross sections')
    require(np.allclose(registration[['pos_x', 'pos_y']], xy, rtol=0, atol=1e-5),
            'Registration and matrix positions disagree')
    require(np.allclose(registration[['x_CCF', 'y_CCF', 'z_CCF']], matrix['CCF'][keep], rtol=0, atol=1e-8, equal_nan=True),
            'Registration and matrix CCF coordinates disagree')

    #region names to CCF IDs (0 = no region; 997 = root, also not a region)
    tree = structure_tree()
    id_by_name = {node['name']: node_id for node_id, node in tree.items()}
    unmapped = set(registration['name']) - set(id_by_name) - {'', 'background', 'Background', 'not found', 'root', 'empty'}
    require(not unmapped, f'Unmapped released region names: {sorted(unmapped)}')
    region_ids = registration['name'].map(id_by_name).fillna(0).to_numpy(np.int64)
    valid = (region_ids != 0) & (region_ids != 997)
    cells['released_region_name'] = registration['name']
    cells['reference_valid'] = valid

    #reference labels for the report: CCF ancestors at depth 5 and 7 (root = 1), and the released regions
    n_regions = {}
    for column, depth in [('reference_depth5', 5), ('reference_depth7', 7), ('reference_released', None)]:
        ancestor = {region: int(region) if depth is None else tree[region]['path'][min(depth, len(tree[region]['path'])) - 1]
                    for region in np.unique(region_ids[valid])}
        populated = set(ancestor.values())
        require(len({tree[region]['acronym'] for region in populated}) == len(populated), f'Repeated acronym in {column}')
        cells[column] = [tree[ancestor[region]]['acronym'] if ok else None for region, ok in zip(region_ids, valid)]
        n_regions[column] = len(populated)

    #the pinned release
    require(len(matrix['id']) == 2167762 and len(labels) == 1259256 and len(cells) == 1161387,
            'Pinned cohort inventory changed')
    require(cells.section.nunique() == 40 and len(genes) == 106 and cells.cell_subclass.nunique() == 46,
            'Pinned sections, genes or cell types changed')
    require(n_regions == {'reference_depth5': 63, 'reference_depth7': 252, 'reference_released': 564},
            f'Pinned reference region counts changed: {n_regions}')
    require((cells.total_counts > 0).all() and int(valid.sum()) == 1150067, 'Pinned counts or reference cells changed')
    require(not cells.cell_id.duplicated().any(), 'Duplicate cell IDs')

    #write the counts, then the metadata last (it marks a finished conversion)
    out.mkdir(parents=True, exist_ok=True)
    expression = ad.AnnData(X=counts, obs=pd.DataFrame(index=cells.cell_id.to_numpy()),
                            var=pd.DataFrame({'gene_symbol': genes}, index=genes))
    expression.write_h5ad(out / 'expression_raw.h5ad', compression='gzip')
    cells.to_parquet(out / 'cells.parquet', index=False)

    #summary
    print(f'{len(cells):,} cells x {len(genes)} genes in {cells.section.nunique()} sections, '
          f'{cells.cell_subclass.nunique()} cell types', flush=True)
    print(f'{n_unmatched:,} published labels have no released counts and are left out', flush=True)
    print(f'{int(valid.sum()):,} cells with a reference region; regions per level: '
          + ', '.join(f'{column} {n}' for column, n in n_regions.items()), flush=True)


def main(root):
    manifest = json.loads((HERE / 'DOWNLOAD_MANIFEST.json').read_text())
    raw = Path(root) / 'data/raw'
    for spec in manifest['files']:
        print(f'Download/checksum: {spec["name"]} ({spec["size"] / 1e9:.3f} GB)', flush=True)
        fetch(spec, raw / spec['name'])
    out = Path(root) / 'data/converted'
    if (out / 'cells.parquet').exists():
        print(f'Converted input exists, skipping: {out}', flush=True)
    else:
        print(f'Converting to notebook input: {out}', flush=True)
        convert(raw, out)
    print('BARSEQ PUBLIC DATA DOWNLOAD AND CONVERSION: PASSED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root.resolve())
