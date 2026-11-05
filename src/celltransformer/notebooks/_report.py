"""Dataset-independent CellTransformer report rendering. No model fitting.

generate_report takes per-cell metadata and saved domain labels that share one row
order (reporting.report_from_notebook establishes that order) and writes a
self-contained review.html, its figures and tables, and results.zip.
"""
from dataclasses import dataclass
from pathlib import Path
import base64
import colorsys
import hashlib
import html
import json
import importlib.metadata
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial import cKDTree


GOLDEN = .61803398875      #hue step between consecutive colors
MISSING_GREY = '#dddddd'   #cells without a domain or reference label
UNDEFINED_GREY = '#bdbdbd' #undefined correlations in the heatmap


@dataclass
class Reference:
    """One reference annotation: per-cell integer codes into names (-1 = no label)."""
    name: str
    codes: np.ndarray
    names: list
    display_name: str = 'Reference'
    ids: list | None = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + str(message), flush=True)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')


def esc(value):
    return html.escape(str(value), quote=True)


def encoded(path):
    """A PNG as a data URI, so the HTML needs no separate image files."""
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def safe_id(value):
    """Short filename-safe ID for a section, reference or method name."""
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def palette(k):
    """k domain colors (RGB): golden-ratio hues with alternating saturation and brightness."""
    return [colorsys.hsv_to_rgb((i * GOLDEN) % 1, .58 + (i % 3) * .12, .75 + (i % 2) * .2) for i in range(k)]


def reference_palette(references):
    """Hex color per label of each reference; a label ID gets the same color in every reference."""
    keys = sorted({str(v) for ref in references.values() for v in (ref.ids or ref.names)})
    assigned = {}
    used = {MISSING_GREY}
    slot = 0
    for key in keys:
        #walk the hue sequence until a color not used yet comes up
        while True:
            rgb = colorsys.hsv_to_rgb((.13 + slot * GOLDEN) % 1, [.68, .84, .95][slot % 3], [.80, .94][slot % 2])
            color = '#%02x%02x%02x' % tuple(round(255 * x) for x in rgb)
            slot += 1
            if color not in used:
                break
        assigned[key] = color
        used.add(color)
    return {name: [assigned[str(v)] for v in (ref.ids or ref.names)] for name, ref in references.items()}


def proportions(labels, types, n_labels, n_types):
    """Cell-type proportions per label (rows sum to 1, empty labels stay 0) and the label sizes."""
    #count every (label, type) pair with one bincount
    pairs = labels.astype(np.int64) * n_types + types
    counts = np.bincount(pairs, minlength=n_labels * n_types).reshape(n_labels, n_types)
    sizes = counts.sum(axis=1)
    fractions = np.divide(counts, sizes[:, None], out=np.zeros_like(counts, dtype=float), where=sizes[:, None] > 0)
    return fractions, sizes


def pearson_rows(a, b):
    """Pearson r between every row of a and every row of b; NaN where a row is constant."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    r = np.divide(a @ b.T, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
    return np.clip(r, -1, 1)


def ordered_rows(values):
    """Row order from average-linkage clustering of the rows (Euclidean distance)."""
    require(np.isfinite(values).all(), 'Composition proportions must be finite')
    if len(values) <= 1:
        return np.arange(len(values))
    return leaves_list(linkage(values, method='average', metric='euclidean'))


def plot_correspondence(correlation, reference_composition, domain_composition,
                        reference_name, path, *, title=None):
    """Save the heatmap of Pearson r (reference regions x domains) and return its ordering.

    Rows and columns are ordered independently by their cell-type compositions;
    r is shown on a fixed -1 to +1 scale, undefined values in grey."""
    #the three tables must describe the same regions, domains and cell types
    require(correlation.index.tolist() == reference_composition.index.tolist(),
            'Reference composition/correlation row labels differ')
    require(correlation.columns.tolist() == domain_composition.index.tolist(),
            'Domain composition/correlation labels differ')
    require(reference_composition.columns.tolist() == domain_composition.columns.tolist(),
            'Cell-type vocabularies differ')
    values = correlation.to_numpy(float)
    require(not np.isinf(values).any(), 'Correlation table contains infinity')
    require(np.all(np.abs(values[np.isfinite(values)]) <= 1.0000001), 'Pearson correlations must lie within [-1,1]')

    row_order = ordered_rows(reference_composition.to_numpy(float))
    column_order = ordered_rows(domain_composition.to_numpy(float))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(UNDEFINED_GREY)

    font = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 11}
    with plt.rc_context(font):
        fig, ax = plt.subplots(figsize=(6.5, 5.3))
        fig.subplots_adjust(left=.16, right=.98, top=.87, bottom=.29)
        image = ax.imshow(np.ma.masked_invalid(values[row_order][:, column_order]), cmap=cmap, vmin=-1, vmax=1,
                          aspect='auto', interpolation='nearest')
        ax.set_title(title or f'{reference_name}–CT2 correspondence', fontweight='bold', pad=18)
        ax.set_ylabel(f'{reference_name} regions (n = {values.shape[0]:,})')
        ax.set_xlabel(f'CT2 domains (n = {values.shape[1]:,})')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(.5)
            spine.set_color('#444444')

        #colorbar below the heatmap
        cax = fig.add_axes([.36, .145, .32, .027])
        bar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[-1, 1])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, pad=2)
        fig.text(.705, .15, 'Pearson r', ha='left', va='center', fontsize=10)

        fig.savefig(path, dpi=200, facecolor='white')
        plt.close(fig)

    return {
        'reference_order': correlation.index[row_order].tolist(),
        'domain_order': correlation.columns[column_order].tolist(),
        'cell_types': reference_composition.columns.tolist(),
        'ordering': 'Independent average-linkage clustering of cell-type proportions, Euclidean distance',
        'scale': [-1, 1],
        'undefined_correlations': int(np.isnan(values).sum()),
        'undefined_color': UNDEFINED_GREY,
        'reference_regions': values.shape[0],
        'requested_domains': values.shape[1],
    }


def boundaries(xy, codes, bounds, grid_size, max_distance):
    """Line segments between reference regions inside bounds, drawn as outlines on enlargements.

    Each grid point takes the label of its nearest labelled cell (if within max_distance);
    a segment is drawn wherever two neighboring grid points differ."""
    lo = np.asarray(bounds[0])
    hi = np.asarray(bounds[1])
    require(grid_size > 0 and max_distance > 0, 'Boundary distances must be positive')
    gx = np.arange(lo[0], hi[0] + grid_size, grid_size)
    gy = np.arange(lo[1], hi[1] + grid_size, grid_size)
    require(len(gx) * len(gy) <= 5_000_000, 'Enlargement boundary raster is too large; increase boundary_grid_size')
    xx, yy = np.meshgrid(gx, gy)
    known = codes >= 0
    if not known.any():
        return np.empty((0, 2, 2))

    #label of the nearest labelled cell at each grid point
    distance, nearest = cKDTree(xy[known]).query(np.c_[xx.ravel(), yy.ravel()])
    grid = codes[known][nearest].reshape(xx.shape)
    valid = (distance <= max_distance).reshape(xx.shape)

    half = grid_size / 2
    edges = []
    #label changes between vertically adjacent points: horizontal segments
    rows, cols = np.nonzero((grid[:-1] != grid[1:]) & valid[:-1] & valid[1:])
    edges.extend([[[gx[i] - half, gy[j] + half], [gx[i] + half, gy[j] + half]] for j, i in zip(rows, cols)])
    #label changes between horizontally adjacent points: vertical segments
    rows, cols = np.nonzero((grid[:, :-1] != grid[:, 1:]) & valid[:, :-1] & valid[:, 1:])
    edges.extend([[[gx[i] + half, gy[j] - half], [gx[i] + half, gy[j] + half]] for j, i in zip(rows, cols)])
    return np.asarray(edges)


def _check_metadata(metadata, section_column, x_column, y_column, cell_id_column):
    """Validate the per-cell metadata and return the (n, 2) coordinates."""
    require(isinstance(metadata, pd.DataFrame) and len(metadata) > 0, 'Supply nonempty per-cell metadata')
    required = {section_column, x_column, y_column, cell_id_column}
    require(required <= set(metadata.columns), f'Missing metadata columns: {required - set(metadata.columns)}')
    cell_ids = metadata[cell_id_column]
    require(cell_ids.notna().all() and not cell_ids.astype(str).duplicated().any(), 'Cell IDs must be present and unique')
    require(metadata[section_column].notna().all(), 'Section names must be present')
    xy = metadata[[x_column, y_column]].to_numpy(float)
    require(np.isfinite(xy).all(), 'Coordinates must be finite')
    return xy


def _check_domain_labels(domain_labels, n):
    """Validate {method: {k: labels}} (codes 0..k-1, -1 unassigned) and return the sorted k values."""
    require(bool(domain_labels), 'No domain assignments supplied')
    ks = sorted({int(k) for assignments in domain_labels.values() for k in assignments})
    require(all(k > 0 for k in ks), 'k must be positive')
    for method, assignments in domain_labels.items():
        for k, labels in assignments.items():
            require(isinstance(k, (int, np.integer)), 'Use integer k keys in domain_labels')
            require(np.shape(labels) == (n,) and np.issubdtype(np.asarray(labels).dtype, np.integer),
                    f'Invalid label array: {method}, k={k}')
            require(np.min(labels) >= -1 and np.max(labels) < k, f'Out-of-range labels: {method}, k={k}')
    return ks


def _check_references(references, n):
    for name, ref in references.items():
        require(np.shape(ref.codes) == (n,) and np.issubdtype(np.asarray(ref.codes).dtype, np.integer),
                f'Invalid reference codes: {name}')
        require(len(ref.names) > 0 and len(set(map(str, ref.names))) == len(ref.names),
                f'Reference names must be unique: {name}')
        require(np.min(ref.codes) >= -1 and np.max(ref.codes) < len(ref.names), f'Out-of-range reference codes: {name}')
        require(ref.ids is None or len(ref.ids) == len(ref.names), 'Reference ID/name lengths differ')


def _references_per_k(reference_for_k, references, ks):
    """{k: reference_id}; a single reference is used for every k."""
    if reference_for_k is None:
        require(len(references) <= 1, 'With multiple references, explicitly supply reference_for_k')
        reference_for_k = {k: next(iter(references)) for k in ks} if references else {}
    else:
        reference_for_k = {int(k): value for k, value in reference_for_k.items()}
    require(set(reference_for_k) <= set(ks) and all(v in references for v in reference_for_k.values()),
            'Invalid reference_for_k mapping')
    return reference_for_k


def _check_crop_bounds(crop_bounds, section_names):
    for section, bounds in crop_bounds.items():
        b = np.asarray(bounds, float)
        require(section in section_names and b.shape == (2, 2) and np.isfinite(b).all() and (b[1] > b[0]).all(),
                f'Invalid enlargement for {section}')


def _training_section(training_history, figures, tables):
    """Save the training curves and their table; return them as HTML."""
    history = pd.DataFrame(training_history if training_history is not None else [])
    if history.empty:
        return '<p>Training history was not supplied.</p>'
    require('epoch' in history, 'Training history requires an epoch column')

    fig, ax = plt.subplots(figsize=(7, 3))
    for column, label in [('training_nll', 'Training NLL'), ('validation_nll', 'Validation NLL')]:
        if column in history:
            history[column] = pd.to_numeric(history[column], errors='coerce')
            ax.plot(history.epoch, history[column], 'o-', label=label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel='Completed training epoch', ylabel='Mean negative log-likelihood')
    if ax.lines:
        ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(figures / 'training.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    history.to_csv(tables / 'training_history.tsv', sep='\t', index=False)
    table = history.to_html(index=False, float_format=lambda v: f'{v:.6f}', na_rep='unavailable')
    return '<img src="' + encoded(figures / 'training.png') + '" alt="Training diagnostics">' + table


def _section_maps(*, xy, section_values, section_names, ks, domain_labels, method_names, eligible,
                  references, reference_for_k, reference_colors, crop_bounds, crop_name,
                  boundary_grid_size, boundary_max_distance, coordinate_units, invert_y, figures):
    """Plot every section at every k: the reference (if any), then one panel per method.

    Sections with crop bounds also get an enlargement of the same figure. Returns the
    encoded images and reference legends by k and section, and one row per saved image."""
    maps, details, legends, rows = {}, {}, {}, []
    for k in ks:
        key = str(k)
        maps[key], details[key], legends[key] = {}, {}, {}
        ref_id = reference_for_k.get(k)
        ref = references.get(ref_id)
        methods = [method for method in domain_labels if k in domain_labels[method]]
        domain_colors = np.asarray(palette(k))

        for section in section_names:
            cells = np.flatnonzero(section_values == section)
            points = xy[cells]
            n_panels = len(methods) + int(ref is not None)
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharex=True, sharey=True, squeeze=False)
            axes = axes[0]
            method_axes = axes
            legend = []

            #reference panel, and its legend with cell counts
            if ref is not None:
                codes = ref.codes[cells]
                known = codes >= 0
                colors = np.full(len(cells), MISSING_GREY, dtype='<U7')
                colors[known] = np.asarray(reference_colors[ref_id])[codes[known]]
                axes[0].scatter(points[:, 0], points[:, 1], c=colors, s=.7, linewidths=0, rasterized=True)
                axes[0].set_title(ref.name)
                method_axes = axes[1:]
                for code in np.unique(codes[known]):
                    color = reference_colors[ref_id][code]
                    count = int((codes == code).sum())
                    legend.append(f'<span class="legend-entry"><i style="background:{color}"></i>'
                                  f'{esc(ref.names[code])} ({count:,})</span>')

            #one panel per method; grey = no domain or no valid neighborhood
            for ax, method in zip(method_axes, methods):
                labels = domain_labels[method][k][cells]
                ok = (labels >= 0) & eligible[cells]
                ax.scatter(points[~ok, 0], points[~ok, 1], c=MISSING_GREY, s=.7, linewidths=0)
                ax.scatter(points[ok, 0], points[ok, 1], c=domain_colors[labels[ok]], s=.7, linewidths=0,
                           rasterized=True)
                ax.set_title(method_names.get(method, method))

            for ax in axes:
                ax.set_aspect('equal')
                ax.set_xlabel(f'x ({coordinate_units})')
            axes[0].set_ylabel(f'y ({coordinate_units})')
            if invert_y:
                axes[0].invert_yaxis()

            #whole section
            fig.suptitle(f'{section} · k = {k} · {len(cells):,} cells', fontsize=12)
            path = figures / f'section_{safe_id(section)}_k{k}.png'
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            maps[key][section] = encoded(path)
            legends[key][section] = ''.join(legend)
            reference_name = ref.name if ref is not None else ''
            rows.append(dict(section=section, k=k, reference=reference_name, view='whole_section', file=path.name))
            details[key][section] = None

            #enlargement: the same figure zoomed to the crop bounds, with optional region outlines
            if section in crop_bounds:
                lo, hi = np.asarray(crop_bounds[section])
                edges = []
                if ref is not None and boundary_grid_size is not None and boundary_max_distance is not None:
                    edges = boundaries(points, ref.codes[cells], crop_bounds[section],
                                       boundary_grid_size, boundary_max_distance)
                for ax in axes:
                    ax.set_xlim(lo[0], hi[0])
                    ax.set_ylim((hi[1], lo[1]) if invert_y else (lo[1], hi[1]))
                    if len(edges):
                        ax.add_collection(LineCollection(edges, colors='black', linewidths=.35, alpha=.4))
                fig.suptitle(f'{section} · {crop_name} · k = {k}', fontsize=12)
                path = figures / f'section_{safe_id(section)}_k{k}_detail.png'
                fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
                details[key][section] = encoded(path)
                rows.append(dict(section=section, k=k, reference=reference_name, view='enlargement', file=path.name))

            plt.close(fig)
            log(f'Spatial map: {section}, k={k}')
    return maps, details, legends, rows


def _write_color_tables(output_dir, references, reference_colors, ks):
    reference_rows = [dict(reference=key, id=str((ref.ids or ref.names)[i]), label=str(name), color=reference_colors[key][i])
                      for key, ref in references.items() for i, name in enumerate(ref.names)]
    pd.DataFrame(reference_rows, columns=['reference', 'id', 'label', 'color']).to_csv(
        output_dir / 'reference_colors.tsv', sep='\t', index=False)
    domain_rows = [dict(k=k, domain=i, color=to_hex(c)) for k in ks for i, c in enumerate(palette(k))]
    pd.DataFrame(domain_rows).to_csv(output_dir / 'domain_colors.tsv', sep='\t', index=False)


def _correspondence(*, metadata, cell_type_column, eligible, references, reference_for_k, domain_labels,
                    method_names, notes, figures, tables):
    """Pearson r between the cell-type compositions of reference regions and domains, per k and method.

    Uses the cells that are eligible and have a reference label, a domain in every method at
    this k, and a cell type. Saves heatmaps and tables; returns {k: {method: panel}} for the HTML.
    A panel that can't be computed adds a note instead."""
    panels = {}
    for k, ref_id in reference_for_k.items():
        ref = references[ref_id]
        methods = [method for method in domain_labels if k in domain_labels[method]]

        #cells shared by every method at this k
        common = eligible & (ref.codes >= 0)
        for method in methods:
            common &= domain_labels[method][k] >= 0
        type_codes = None
        type_names = None
        if cell_type_column is not None and cell_type_column in metadata:
            present = metadata[cell_type_column].notna().to_numpy()
            common &= present
            type_names = sorted(metadata.loc[present, cell_type_column].astype(str).unique())
            type_codes = pd.Categorical(metadata[cell_type_column].astype(str), categories=type_names).codes

        panels[str(k)] = {}
        for method in methods:
            method_name = method_names.get(method, method)
            if type_codes is None or not common.any():
                notes.append(f'Correspondence unavailable for {ref.name}, k={k}, {method_name}: '
                             'no usable cell-type annotations.')
                continue

            #compositions and their correlation; empty regions or domains are undefined
            reference_props, reference_sizes = proportions(ref.codes[common], type_codes[common],
                                                           len(ref.names), len(type_names))
            domain_props, domain_sizes = proportions(domain_labels[method][k][common], type_codes[common],
                                                     k, len(type_names))
            values = pearson_rows(reference_props, domain_props)
            values[reference_sizes == 0, :] = np.nan
            values[:, domain_sizes == 0] = np.nan
            correlation = pd.DataFrame(values, index=list(map(str, ref.names)), columns=[f'D{i}' for i in range(k)])
            reference_composition = pd.DataFrame(reference_props, index=correlation.index, columns=type_names)
            domain_composition = pd.DataFrame(domain_props, index=correlation.columns, columns=type_names)

            #heatmap, its ordering, and the tables behind it
            token = f'correspondence_{safe_id(ref_id)}_{safe_id(method)}_k{k}'
            path = figures / (token + '.png')
            order = plot_correspondence(correlation, reference_composition, domain_composition, ref.display_name, path)
            n_cells = int(common.sum())
            order.update(reference=ref.name, method=method_name, k=k,
                         source='Computed from supplied reference labels, saved domain assignments '
                                'and cell-type labels on common eligible cells',
                         evaluation_cells=n_cells)
            write_json(tables / (token + '_order.json'), order)
            correlation.to_csv(tables / (token + '.tsv'), sep='\t')
            reference_composition.to_csv(tables / (token + '_reference_proportions.tsv'), sep='\t')
            domain_composition.to_csv(tables / (token + '_domain_proportions.tsv'), sep='\t')

            panels[str(k)][method] = {
                'image': encoded(path),
                'reference': ref.name,
                'method': method_name,
                'undefined': order['undefined_correlations'],
                'n_reference': len(reference_composition),
                'k': k,
                'n_cells': n_cells,
            }
            log(f'Correspondence: {ref.name}, {method_name}, k={k}')
    return panels


CSS = '''
body{font:16px/1.55 system-ui,sans-serif;color:#21323d;background:#f5f7fa;margin:0}
main{max-width:1500px;margin:auto;padding:32px}
h1,h2{line-height:1.2}
h1{font-size:36px}
section{background:white;border:1px solid #dce2e8;border-radius:12px;padding:24px;margin:24px 0}
img{max-width:100%;height:auto}
pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}
table{border-collapse:collapse;font-size:12px}
td,th{padding:7px;border:1px solid #dde2e8}
select,button{font:inherit;padding:8px;margin:5px}
.note{background:#fff5d9;padding:15px;border-left:4px solid #cca64b}
.scroll{overflow:auto}
.small{font-size:13px;color:#536773}
.badges{font-size:20px;color:#147783}
.legend-entry{display:inline-block;margin:5px 12px 5px 0}
.legend-entry i{display:inline-block;width:14px;height:14px;border:1px solid #aaa;margin-right:4px}
#correspondence-image{width:750px;max-width:100%}
'''

#page behavior: section/k/view menus for the maps, k/method menus for the heatmaps; reads `data`
SCRIPT = '''
const s = document.getElementById('section'), k = document.getElementById('k'),
      view = document.getElementById('view'), map = document.getElementById('map');

function showMap() {
    const detail = data.details[k.value][s.value];
    map.src = view && view.value === 'detail' && detail ? detail : data.maps[k.value][s.value];
    document.getElementById('viewnote').textContent = view && view.value === 'detail' && !detail
        ? 'No enlargement was supplied for this section; showing the whole section.' : '';
    const legend = document.getElementById('legend');
    if (legend) legend.innerHTML = data.legends[k.value][s.value];
}

k.value = k.options[k.options.length - 1].value;
s.onchange = showMap;
k.onchange = showMap;
if (view) view.onchange = showMap;
document.getElementById('zoom').onclick = () => map.style.maxWidth = map.style.maxWidth === 'none' ? '100%' : 'none';
showMap();

const ck = document.getElementById('correspondence-k'), cm = document.getElementById('correspondence-method');

function showCorrespondence() {
    const v = data.correspondence[ck.value][cm.value];
    document.getElementById('correspondence-image').src = v.image;
    document.getElementById('correspondence-note').textContent = v.reference + ' · ' + v.method + ' · '
        + v.n_reference + ' reference regions · ' + v.k + ' requested domains · '
        + v.n_cells.toLocaleString() + ' evaluation cells · ' + v.undefined + ' undefined correlations';
}

function chooseMethods() {
    const previous = cm.value;
    cm.replaceChildren();
    for (const [id, v] of Object.entries(data.correspondence[ck.value])) {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = v.method;
        cm.appendChild(option);
    }
    //keep the previous method if it exists at this k, else prefer the smoothed one
    if ([...cm.options].some(o => o.value === previous)) cm.value = previous;
    else {
        const smoothed = [...cm.options].find(o => o.textContent.includes('+smoothing'));
        if (smoothed) cm.value = smoothed.value;
    }
    showCorrespondence();
}

if (ck) {
    ck.onchange = chooseMethods;
    cm.onchange = showCorrespondence;
    chooseMethods();
}
'''


def _render_html(*, title, n_cells, section_names, ks, n_genes, notes, training_html, references, crop_bounds,
                 crop_name, boundary_grid_size, boundary_max_distance, coordinate_units, correspondence,
                 use_cell_type, configuration_text, provenance, payload):
    """The self-contained review page: all images are inlined in payload."""
    units = esc(coordinate_units)
    parts = []

    #header, notes and training
    parts.append('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                 '<meta name="viewport" content="width=device-width,initial-scale=1">'
                 f'<title>{esc(title)}</title><style>{CSS}</style></head><body><main>')
    gene_text = f' · {int(n_genes):,} genes' if n_genes is not None else ''
    parts.append(f'<h1>{esc(title)}</h1><p class="badges">{n_cells:,} cells · {len(section_names)} sections{gene_text}</p>')
    if notes:
        parts.append('<div class="note"><ul>' + ''.join(f'<li>{esc(note)}</li>' for note in notes) + '</ul></div>')
    parts.append('<section><h2>Training diagnostics</h2><div class="scroll">' + training_html + '</div>'
                 '<p>Curves show supplied epoch-averaged losses. Missing values remain undefined. Lower validation '
                 'loss does not by itself establish better anatomical recovery.</p></section>')

    #spatial domain maps
    section_options = ''.join(f'<option value="{esc(s)}">{esc(s)}</option>' for s in section_names)
    k_options = ''.join(f'<option value="{k}">k = {k}</option>' for k in ks)
    parts.append('<section><h2>Spatial domains</h2><p>Each dot represents a cell. Section names come directly from '
                 'the supplied metadata. Colors identify categories consistently across sections within each method '
                 'and k; matching colors across independent clusterings or the reference do not imply corresponding '
                 'identities. Grey indicates an unavailable assignment or an excluded model center.</p>')
    parts.append(f'<label>Section <select id="section">{section_options}</select></label>'
                 f'<label>Clustering resolution <select id="k">{k_options}</select></label>')
    if crop_bounds:
        parts.append('<label>View <select id="view"><option value="full">Whole section</option>'
                     f'<option value="detail">{esc(crop_name)}</option></select></label>')
    parts.append('<button id="zoom">Toggle full-size view</button><p id="viewnote" class="small"></p>'
                 '<div class="scroll"><img id="map" alt="Spatial domain maps"></div>')
    if references:
        parts.append('<details><summary>Reference labels and cell counts in this section</summary>'
                     '<div id="legend"></div></details>')
    parts.append(f'<p class="small">Coordinates are displayed in {units}. Every supplied section is included.</p>')
    if crop_bounds:
        parts.append(f'<p class="small">{esc(crop_name)} uses the supplied section-specific crop bounds with the same '
                     'limits across methods.')
        if boundary_grid_size is not None and boundary_max_distance is not None:
            parts.append(f' Reference outlines are inferred from nearby cell labels on a {boundary_grid_size:g}-{units} '
                         f'grid within {boundary_max_distance:g} {units} of a labelled cell. They are visual guides, '
                         'not original atlas contours.')
        parts.append('</p>')
    parts.append('</section>')

    #reference-domain correspondence
    if correspondence:
        options = ''.join(f'<option value="{k}">k = {k}</option>' for k in correspondence)
        parts.append('<section><h2>Reference–domain correspondence</h2><p>Pearson correlation between the cell-type '
                     'proportions of each reference region and each discovered domain. Rows and columns are ordered '
                     'independently by average-linkage clustering of those proportions. Colors range from −1 (blue) to '
                     '+1 (red); grey denotes an undefined correlation. This measures similarity in cell-type '
                     'composition, not direct spatial overlap or a one-to-one anatomical match.</p>')
        parts.append(f'<label>Clustering resolution <select id="correspondence-k">{options}</select></label>'
                     '<label>Embedding variant <select id="correspondence-method"></select></label>'
                     '<p id="correspondence-note" class="small"></p>'
                     '<img id="correspondence-image" alt="Reference–domain Pearson correlation heatmap"></section>')

    #run details
    if use_cell_type:
        cell_type_sentence = ('Cell-type annotations used for the model also contribute to composition-based '
                              'correspondence; the reference is an external comparison when it was not used during '
                              'training.')
    else:
        cell_type_sentence = ('The model was trained without cell-type annotations (model.use_cell_type=false), so no '
                              'composition-based correspondence is computed.')
    parts.append('<section><h2>Run details and provenance</h2><p>The report reads saved domain assignments. It does '
                 'not train a model, extract embeddings, smooth embeddings or fit new clusters. '
                 + cell_type_sentence + '</p>'
                 '<details><summary>Analysis configuration</summary><pre>' + esc(configuration_text) + '</pre></details>'
                 '<details><summary>Source information</summary><pre>'
                 + esc(json.dumps(provenance, indent=2, default=str)) + '</pre></details>'
                 '<p>The companion results.zip contains this self-contained HTML, figures, correspondence tables, '
                 'matrix ordering, palettes and provenance. Large input arrays and checkpoints are not copied into '
                 'the archive.</p></section>')

    #images and legends as JSON (escaped so it can't close the script tag), then the page script
    data = json.dumps(payload).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    parts.append(f'<script>const data={data};{SCRIPT}</script></main></body></html>')
    return ''.join(parts)


def generate_report(*, metadata, domain_labels, output_dir, dataset_name='Spatial transcriptomics dataset',
                    section_column='section', x_column='x', y_column='y', cell_id_column='cell_id',
                    cell_type_column='cell_type', use_cell_type=True, eligible=None, references=None, reference_for_k=None,
                    method_names=None, coordinate_units='µm', training_history=None, n_genes=None,
                    notes=None, provenance=None, crop_bounds=None, crop_name='Enlargement',
                    boundary_grid_size=None, boundary_max_distance=None, invert_y=True):
    """Render aligned, already-computed domain assignments; references are optional.

    domain_labels: {method: {k: integer array}}, codes 0..k-1; -1 unassigned.
    references: {reference_id: Reference(...)}, codes 0..n_regions-1; -1 missing.
    reference_for_k: {k: reference_id}; a single reference is reused for all k.
    crop_bounds: optional {section: [[xmin,ymin],[xmax,ymax]]}, in coordinate units.
    Training history is a list/DataFrame with epoch, training_nll, validation_nll.
    use_cell_type=False (model trained without cell types): no composition heatmaps,
    even if the metadata has cell-type labels.
    """
    #check all inputs before writing anything
    output_dir = Path(output_dir)
    require(not output_dir.exists(), 'Choose a new report directory; previous reports are preserved')
    xy = _check_metadata(metadata, section_column, x_column, y_column, cell_id_column)
    section_values = metadata[section_column].astype(str).to_numpy()
    section_names = sorted(set(section_values))
    n = len(metadata)
    eligible = np.ones(n, bool) if eligible is None else np.asarray(eligible)
    require(eligible.dtype == bool and eligible.shape == (n,), 'Eligible mask must be one Boolean per cell')
    references = references or {}
    method_names = method_names or {}
    crop_bounds = crop_bounds or {}
    notes = list(notes or [])
    provenance = provenance or {}
    if not use_cell_type:
        #no cell types in the model -> no composition heatmaps
        cell_type_column = None
        notes.append('The model was trained without cell types (model.use_cell_type=false), '
                     'so cell-type composition heatmaps are not shown.')
    ks = _check_domain_labels(domain_labels, n)
    methods = list(domain_labels)
    _check_references(references, n)
    reference_for_k = _references_per_k(reference_for_k, references, ks)
    _check_crop_bounds(crop_bounds, section_names)

    #output folders and provenance
    output_dir.mkdir(parents=True)
    figures = output_dir / 'figures'
    figures.mkdir()
    tables = output_dir / 'tables'
    tables.mkdir()
    provenance = dict(provenance)
    provenance['report_renderer'] = {
        'version': '2',
        'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'software': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'matplotlib']},
    }
    write_json(output_dir / 'provenance.json', provenance)

    #figures and tables
    training_html = _training_section(training_history, figures, tables)
    reference_colors = reference_palette(references)
    maps, details, legends, image_rows = _section_maps(
        xy=xy, section_values=section_values, section_names=section_names, ks=ks, domain_labels=domain_labels,
        method_names=method_names, eligible=eligible, references=references, reference_for_k=reference_for_k,
        reference_colors=reference_colors, crop_bounds=crop_bounds, crop_name=crop_name,
        boundary_grid_size=boundary_grid_size, boundary_max_distance=boundary_max_distance,
        coordinate_units=coordinate_units, invert_y=invert_y, figures=figures)
    pd.DataFrame(image_rows).to_csv(output_dir / 'section_images.tsv', sep='\t', index=False)
    _write_color_tables(output_dir, references, reference_colors, ks)
    correspondence = {}
    if use_cell_type:
        correspondence = _correspondence(
            metadata=metadata, cell_type_column=cell_type_column, eligible=eligible, references=references,
            reference_for_k=reference_for_k, domain_labels=domain_labels, method_names=method_names,
            notes=notes, figures=figures, tables=tables)
    #k values with at least one heatmap
    correspondence = {k: panels for k, panels in correspondence.items() if panels}
    n_panels = sum(len(panels) for panels in correspondence.values())

    #configuration, written after the correspondence step so it includes that step's notes
    configuration = dict(
        dataset=dataset_name, section_column=section_column, x_column=x_column, y_column=y_column,
        coordinate_units=coordinate_units, ks=ks, methods=methods, reference_for_k=reference_for_k,
        crop_bounds=crop_bounds, notes=notes, n_cells=n, eligible_centers=int(eligible.sum()),
        references={key: {'name': ref.name, 'display_name': ref.display_name, 'names': ref.names, 'ids': ref.ids}
                    for key, ref in references.items()},
        correspondence='Pearson r of cell-type proportions; no domain refitting', use_cell_type=use_cell_type)
    write_json(output_dir / 'report_configuration.json', configuration)

    #the page
    document = _render_html(
        title=f'{dataset_name} · CellTransformer', n_cells=n, section_names=section_names, ks=ks, n_genes=n_genes,
        notes=notes, training_html=training_html, references=references, crop_bounds=crop_bounds,
        crop_name=crop_name, boundary_grid_size=boundary_grid_size, boundary_max_distance=boundary_max_distance,
        coordinate_units=coordinate_units, correspondence=correspondence, use_cell_type=use_cell_type,
        configuration_text=(output_dir / 'report_configuration.json').read_text(), provenance=provenance,
        payload={'maps': maps, 'details': details, 'legends': legends, 'correspondence': correspondence})
    (output_dir / 'review.html').write_text(document)

    #summary and archive
    summary = dict(dataset=dataset_name, cells=n, sections=len(section_names), ks=ks, reference_count=len(references),
                   correspondence_panels=n_panels, section_images=len(image_rows), standalone_html=True)
    write_json(output_dir / 'summary.json', summary)
    with zipfile.ZipFile(output_dir / 'results.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(output_dir.rglob('*')):
            if path.is_file() and path.name != 'results.zip':
                archive.write(path, path.relative_to(output_dir))
    log(f'HTML REPORT COMPLETED: {output_dir / "review.html"}')
    return output_dir / 'review.html'
