"""Optional Step 13 for native notebooks; no unattended-run manifest required."""
from pathlib import Path
import time
import os
import json
import shutil
import tempfile

import numpy as np
import pandas as pd

from ._report import Reference, generate_report, require


def read_reference_table(path, id_column, columns):
    """Load only selected annotation columns; an empty mapping disables comparison."""
    if not columns or path is None:
        return None
    path = Path(path)
    selected = list(dict.fromkeys([id_column, *columns.values()]))
    if path.suffix == '.parquet':
        table = pd.read_parquet(path, columns=selected)
    else:
        # Preserve literal labels such as NA; missing annotations are empty strings.
        table = pd.read_csv(path, usecols=selected, keep_default_na=False,
                            dtype={column: 'category' for column in selected if column != id_column},
                            converters={id_column: str})
    require(table[id_column].notna().all(), 'Reference cell IDs cannot be missing')
    return table


def read_use_cell_type(checkpoint):
    """Whether the checkpoint's model was trained with cell types.

    Checkpoints from before the model.use_cell_type setting count as True.
    """
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return bool(saved['hyper_parameters']['model_cfg'].get('use_cell_type', True))


def read_training_history(checkpoint, expected_epochs=None):
    """Read local epoch averages for this checkpoint's run, without a cloud login.

    W&B's datastore reader opens files r+b, so it reads a temporary copy. Missing
    values remain undefined. Step-level losses are never substituted.
    """
    checkpoint = Path(checkpoint)
    checkpoint_dir = checkpoint.parent
    records = {}; notes = []
    info_path = checkpoint_dir / 'wandb_run.json'
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    candidates = sorted(checkpoint_dir.glob('wandb/offline-run-*/run-*.wandb'))
    if info.get('run_id'):
        candidates = [path for path in candidates if path.name == f"run-{info['run_id']}.wandb"]
    if len(candidates) == 1:
        try:
            from wandb.sdk.internal.datastore import DataStore
            from wandb.proto import wandb_internal_pb2
            with tempfile.TemporaryDirectory(prefix='ct2-report-history-') as temporary:
                copied = Path(temporary) / candidates[0].name
                shutil.copyfile(candidates[0], copied)
                store = DataStore()
                store.open_for_scan(str(copied))
                try:
                    epoch = None
                    while True:
                        data = store.scan_data()
                        if data is None:
                            break
                        record = wandb_internal_pb2.Record()
                        record.ParseFromString(data)
                        if not record.HasField('history'):
                            continue
                        row = {item.key or '/'.join(item.nested_key): json.loads(item.value_json)
                               for item in record.history.item}
                        #pre-training validation pass: recorded as epoch 0 (no training loss exists yet)
                        value = row.get('valid_pretrain/nll_epoch')
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            records[0] = {'epoch': 0, 'training_nll': None, 'validation_nll': float(value)}
                        if 'epoch' in row:
                            epoch = int(row['epoch']) + 1
                        if epoch is None:
                            continue
                        for key, destination in [('train/nll_epoch', 'training_nll'), ('valid/nll_epoch', 'validation_nll')]:
                            if key in row and isinstance(row[key], (int, float)) and np.isfinite(row[key]):
                                records.setdefault(epoch, {'epoch': epoch, 'training_nll': None, 'validation_nll': None})[destination] = float(row[key])
                finally:
                    store.close()
        except Exception as exc:
            notes.append(f'Local training history could not be fully read: {type(exc).__name__}: {exc}')
    else:
        notes.append(f'Expected one local offline history file for the selected checkpoint; found {len(candidates)}. No online history was requested.')
    if expected_epochs is not None:
        for epoch in range(1, int(expected_epochs) + 1):
            records.setdefault(epoch, {'epoch': epoch, 'training_nll': None, 'validation_nll': None})
    missing = {name: [i for i in sorted(records) if records[i][name] is None and not (i == 0 and name == 'training_nll')]
               for name in ['training_nll', 'validation_nll']}
    if any(missing.values()):
        notes.append(f'Missing epoch-average loss values: {missing}. Batch losses are not substituted.')
    return {'history': [records[i] for i in sorted(records)], 'notes': notes,
            'provenance': {'checkpoint': str(checkpoint.resolve()), 'wandb_run': info,
                           'offline_history': str(candidates[0]) if len(candidates) == 1 else None,
                           'epoch_numbering': 'Logged zero-based epoch plus one', 'missing_values': missing}}


def show_report(path):
    """Show lightweight file links; do not embed the entire HTML in notebook outputs."""
    from IPython.display import FileLink, display
    path = Path(path).resolve()
    print('HTML:', path)
    print('Results ZIP:', path.with_name('results.zip'))
    print('If the output is outside the notebook server directory, copy the printed HTML file to your local computer to open it.')
    display(FileLink(os.path.relpath(path, Path.cwd())))
    display(FileLink(os.path.relpath(path.with_name('results.zip'), Path.cwd())))


def report_from_notebook(*, work_dir, embedding_sets, domains_dir, ks,
                         dataset_name='Spatial transcriptomics dataset', method_names=None,
                         reference_metadata=None, reference_columns=None, reference_id_column='cell_id',
                         reference_for_k=None, reference_display_names=None, cell_id_column='cell_id',
                         section_column='section', x_column='x', y_column='y', cell_type_column='cell_type',
                         use_cell_type=True, coordinate_units='µm', training_history=None, n_genes=None,
                         crop_bounds=None, crop_name='Enlargement', notes=None, provenance=None,
                         output_dir=None, boundary_grid_size=None, boundary_max_distance=None,
                         crop_selection=None):
    """Read saved native outputs and call the dataset-independent renderer.

    embedding_sets is the notebook's {tag: embeddings_directory} mapping.
    reference_columns maps a display/reference key to a metadata column, e.g.
    {"Histological annotation": "annotated_region"}. No reference is required.
    For multiple reference levels, provide reference_for_k explicitly.
    training_history may be recorded epoch dictionaries, a DataFrame, or None.
    use_cell_type is read_use_cell_type(checkpoint); when False, cell_type_column
    is not needed and no composition heatmaps are made.
    """
    work_dir=Path(work_dir); domains_dir=Path(domains_dir)
    require(embedding_sets, 'Supply at least one completed embedding set')
    tags=list(embedding_sets); first=Path(embedding_sets[tags[0]])
    files=sorted(first.glob('*_embeddings.npz'))
    require(files, 'No native embedding files found')
    wanted=[path.name for path in files]; frames=[]; masks=[]; inventory=[]
    if not use_cell_type:cell_type_column=None
    needed=[cell_id_column,section_column,x_column,y_column]+([cell_type_column] if cell_type_column else [])
    for path in files:
        metadata_path=path.with_name(path.stem.removesuffix('_embeddings')+'.parquet')
        frame=pd.read_parquet(metadata_path)
        require(set(needed)<=set(frame), f'Missing metadata columns: {metadata_path}')
        with np.load(path,allow_pickle=False) as archive:
            mask=archive['has_neighbors'].copy()
        require(mask.dtype==bool and mask.shape==(len(frame),), f'Invalid neighborhood mask: {path}')
        frames.append(frame); masks.append(mask)
        inventory.append({'embedding_file':str(path.resolve()),'metadata_file':str(metadata_path.resolve()),'cells':len(frame)})
    for tag in tags[1:]:
        directory=Path(embedding_sets[tag]); require([p.name for p in sorted(directory.glob('*_embeddings.npz'))]==wanted,'Embedding section inventories differ')
        for i,name in enumerate(wanted):
            path=directory/name; table=pd.read_parquet(path.with_name(path.stem.removesuffix('_embeddings')+'.parquet'))
            pd.testing.assert_frame_equal(frames[i][needed].reset_index(drop=True),table[needed].reset_index(drop=True),check_dtype=False,check_categorical=False)
            with np.load(path,allow_pickle=False) as archive:
                require(np.array_equal(masks[i],archive['has_neighbors']), 'Neighborhood masks differ across embedding variants')
    metadata=pd.concat(frames,ignore_index=True); eligible=np.concatenate(masks)
    require(not metadata[cell_id_column].astype(str).duplicated().any(), 'Duplicate cell IDs across native sections')
    labels={tag:{int(k):np.load(domains_dir/f'{tag}_kmeans_k{k}_labels.npy',mmap_mode='r',allow_pickle=False) for k in ks} for tag in tags}
    refs={}; reference_columns=reference_columns or {}; reference_display_names=reference_display_names or {}
    aligned = None
    if reference_metadata is not None:
        require(bool(reference_columns),'Specify which reference columns to use')
        source=reference_metadata.copy()
        require(reference_id_column in source,'Reference metadata lacks its cell-ID column')
        require(source[reference_id_column].notna().all(),'Reference cell IDs cannot be missing')
        source[reference_id_column]=source[reference_id_column].astype(str)
        require(not source[reference_id_column].duplicated().any(),'Reference cell IDs must be unique')
        source=source.set_index(reference_id_column)
        aligned=source.reindex(metadata[cell_id_column].astype(str).to_numpy())
        for name,column in reference_columns.items():
            require(column in aligned,f'Missing reference column: {column}')
            values=aligned[column]; present=values.notna() & (values.astype(str)!='')
            names=sorted(values[present].astype(str).unique().tolist())
            require(names,f'No cells have reference labels for {name}')
            codes=np.full(len(metadata),-1,np.int32)
            codes[present.to_numpy()]=pd.Categorical(values[present].astype(str),categories=names).codes
            refs[name]=Reference(name,codes,names,reference_display_names.get(name,name),names)
    elif reference_columns:
        raise ValueError('reference_columns was supplied without reference_metadata')
    if crop_selection is not None:
        require(crop_bounds is None, 'Supply either crop_selection or explicit crop_bounds')
        require(aligned is not None, 'A reference table is needed for annotation-based enlargements')
        preferred = crop_selection['column']
        fallback = crop_selection.get('fallback_column')
        require(preferred in aligned and (fallback is None or fallback in aligned), 'Enlargement columns must be loaded in the reference table')
        preferred_values = aligned[preferred].astype(str).to_numpy()
        fallback_values = aligned[fallback].astype(str).to_numpy() if fallback else None
        prefixes = tuple(crop_selection.get('prefixes', []))
        exact = set(crop_selection.get('values', []))
        minimum = int(crop_selection.get('min_cells', 100))
        padding = float(crop_selection.get('padding', 0))
        crop_bounds = {}
        for section, positions in metadata.groupby(section_column, observed=True).indices.items():
            use = np.array([value in exact or (bool(prefixes) and value.startswith(prefixes)) for value in preferred_values[positions]])
            if use.sum() < minimum and fallback_values is not None:
                use = np.isin(fallback_values[positions], crop_selection.get('fallback_values', []))
            if use.sum() >= minimum:
                coords = metadata.iloc[positions[use]][[x_column, y_column]].to_numpy(float)
                crop_bounds[str(section)] = [(coords.min(0)-padding).tolist(), (coords.max(0)+padding).tolist()]
    notes=list(notes or [])
    if not eligible.all():notes.append(f'{int((~eligible).sum()):,} centers without valid neighborhoods are grey on domain maps and excluded from correspondence calculations. Saved smoothing and clustering are preserved.')
    details=dict(provenance or {})
    details.update(cell_order='Sorted native embedding filenames, then per-section metadata row order',
                   native_files=inventory,work_dir=str(work_dir.resolve()),reference_join='Exact cell ID; unmatched reference annotations remain missing')
    output=Path(output_dir) if output_dir else work_dir/'html_reports'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+f'-{os.getpid()}')
    result=generate_report(metadata=metadata,domain_labels=labels,output_dir=output,dataset_name=dataset_name,
        cell_id_column=cell_id_column,section_column=section_column,x_column=x_column,y_column=y_column,
        cell_type_column=cell_type_column,use_cell_type=use_cell_type,eligible=eligible,references=refs,reference_for_k=reference_for_k,
        method_names=method_names,coordinate_units=coordinate_units,training_history=training_history,n_genes=n_genes,
        notes=notes,provenance=details,crop_bounds=crop_bounds,crop_name=crop_name,
        boundary_grid_size=boundary_grid_size,boundary_max_distance=boundary_max_distance)
    (work_dir/'LATEST_HTML_REPORT').write_text(str(output.resolve())+'\n')
    return result
