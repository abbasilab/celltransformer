import argparse
import os
import warnings
import pathlib

import anndata as ad
import pandas as pd
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder

from brainformr.data import CenterMaskSampler, collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, help="Path to anndata file.", required=True)
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file.", required=True)
    parser.add_argument("--config_path", type=str, help="Path to config file.", required=True)
    parser.add_argument('--patch_size', type=int, help='Patch size for model.', required=False, default=17)
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file from training.', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save embeddings.', required=True)
    parser.add_argument('--bs', type=int, help='Batch size for inference.', required=False, default=32)
    parser.add_argument('--celltype_colname', type=str, help='The column name that contains the cell type assignments', required=False, default='subclass')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.adata_path):
        raise FileNotFoundError(f"Anndata file not found: {args.adata_path}")

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.adata_path)
    # filter control probes
    adata[:, ~adata.var.index.str.contains("Blank")]
    #adata[:, ~adata.var.gene_symbol.str.contains("Blank")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        metadata = pd.read_csv(args.metadata_path)

    cell_type_colname = 'subclass_name' # args.celltype_colname

    metadata["cell_label"] = metadata["cell_label"].astype(str)
    metadata['cell_type'] = metadata[cell_type_colname].astype(str)
    # metadata["x"] = metadata["x_reconstructed"] * 100
    # metadata["y"] = metadata["y_reconstructed"] * 100]
    metadata["x"] = metadata["spatial_x"] / 10.
    metadata["y"] = metadata["spatial_y"] / 10.

    metadata = metadata[
        [
            "cell_type",
            "cell_label",
            "x",
            "y",
            "brain_section_label",
        ]
    ]

    metadata = metadata.sort_values(by='brain_section_label')

    metadata["cell_type"] = LabelEncoder().fit_transform(metadata['cell_type'])
    metadata["cell_type"] = metadata["cell_type"].astype(int)

    metadata = metadata.reset_index(drop=True)

    adata = adata[metadata["cell_label"]]
    patch_size = (args.patch_size, args.patch_size)
    bs = args.batch_size

    sampler = CenterMaskSampler(
        metadata=metadata,
        adata=adata,
        patch_size=patch_size,
        cell_id_colname="cell_label",
        cell_type_colname="cell_type",
        tissue_section_colname="brain_section_label",
        max_num_cells=None, # try setting to a smaller number to run on a smaller GPU
    )

    loader = torch.utils.data.DataLoader(
        sampler,
        batch_size=bs,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        collate_fn=collate,
        prefetch_factor=2,
    )

    config_dct = OmegaConf.load(args.config_path)

    model = instantiate(config_dct.model)
    model = model.cuda()
    model.put_device = "cuda"

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = {}

    model_state_dict = checkpoint['state_dict']
    for key in model_state_dict:
        key_wo_model = key.replace('model.', '')
        state_dict[key_wo_model] = model_state_dict[key]
    # if continuing after Lightning checkpoint need to change keys
    # otherwise can instantiate Lightning object and use that for 
    # embedding generation

    model.load_state_dict(state_dict, strict=True)

    # compiling model didn't seem to improve throughput for me; but ymmv
    # model = torch.compile(model)

    embeds = []

    with torch.inference_mode():
        for batch in tqdm.tqdm(loader):
            fwd = model(batch)
            embed = fwd["neighborhood_repr"].detach().cpu()
            embeds.append(embed)

    embeds = torch.cat(embeds)
    torch.save(embeds, "aibs-embeds.pth")

if __name__ == "__main__":
    main()
