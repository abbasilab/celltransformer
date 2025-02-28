# Getting started

This is documentation for code written as part of the manuscript ["Data-driven fine-grained region discovery in the mouse brain with transformers"](https://www.biorxiv.org/content/10.1101/2024.05.05.592608v1). 

## Installation

* `pip install git+github.com:abbasilab/celltransformer.git` or clone and pip install; alternatively use the Dockerfile -- `PyTorch` and some other heavy libraries are required, which will take more time. It's also a somewhat unoptimized Docker image (no multi-stage build etc.) so caveat emptor.

## Getting started with training on different datasets

Requirements are a CSV file with cell types and cell IDs corresponding to an `anndata` object with probe counts. To set these up for use with the code in this repo:

1. make sure you provide the right cardinality (number of cell types) to the model using the `model` config. You can also do this usinmg the `hydra` CLI by just passing `python [SCRIPT.PY] +model.cell_cardinality=9000` (or whatever the value is)
2. the code right now unfortunately assumes that the values in your anndata.X are `log1p` transformed and then uses `.exp()` in the implementation (`training/lightning_model.py`) to produce counts again for `scvi.NegativeBinomial` -- make sure you follow this convention or edit the training file for a different convention.

### I want to edit the anndata (MERFISH probe counts) and CSV (cell metadata) to work with this codebase

1. Provide in your CSV (case sensitive):

	| column name | description |
	|----------|----------|
	| `cell_type`   | integer encoded class label for the cell type of a given cell   |
	| `cell_label`     | value that will be used to index the `anndata` object. Make sure it is of appropriate datatype because we do not perform any transformation on it (such as conversion to str or int) prior to indexing the `anndata` object. |
	| `brain_section_label`   | value that we will `.groupby()` on to select individual tissue sections to get the cells |
	| `x` | spatial coordinate that will be used to identify neighbors. Must be in same units as `patch_size` argument (default in `hydra` configs is micron). | 
	| `y` | similar as `x` | 

2. Change paths in the `hydra` config file template (`scripts/config/data/mouse1.yaml`). If you do not do this, the script will almost certainly not work.
3. Make sure value of `patch_size` fits your `x` and `y` created in step (1) in the same YAML file.
4. Adjust model parameters in `scripts/config/model/` for desired model. 
5. Set `config_path` in `hydra.main` decorator to the path and config file of interest (see template file `scripts/training/train_base.py`).
6. Set up `wandb` parameters in `scripts/training/train_base.py`.
7. Run with `python train_base.py`.

### I want to edit this at the hydra config level or in the starter script

1. Hydra config file setup: change data paths in `config/data` or create new data yaml file in that folder that has the same fields as the examples in that directory. Make sure to specify:
	- `celltype_colname`: the column that gives the cell type of the cells in the dataset (if you are templating from the `train_aibs_mouse.py` file, which we recommend, we will use `sklearn.preprocessing.LabelEncoder` on this column in the base `train_aibs_mouse.py` file, so it doesn't matter if it's integer or string encoded. If you are not using that function, make sure the dataframe you pass to `CenterMaskSampler` has column `cell_type` (case sensitive default argument) which *must* integer encoded.)
		- NOTE: we are basically assuming you want to train on one mouse, so if there are multiple and there is a chance that one mouse out of multiple has some cell types that are not shared, you need to separately fit the LabelEncoder and then provide integer-encoded class labels (see `train_zhuang.py` for an example.)
	- `cell_id_colname`: the column that gives an ID that we can use to lookup into the h5ad file for single cells' probe count profiles. Make sure this is of the same datatype as the row ID's in the `anndata` object (i.e. make sure that the ID isn't 12345: uint | int instead of 12345: str). 
2. specify model architecture (depth, width etc.) configs in `config/model` yaml file
3. implement the `load_data` method for `BaseTrainer` in `scripts/training/lightning_model.py` (see `train_zhuang.py` and `train_aibs_mouse.py` for reference)
	- in essence this is normalization: mapping spatial x and y coordinates in your data to "x" and "y" and scaling them, also normalizing cell type column names as described in (1), optionally. One example might be to filter control probes. 
		- specifically you can look at `load_data` in `train_aibs_mouse` to see an example, but the dataloader code assumes that the spatial columns are `x` and `y`. 
		- we also don't automatically rescale the units of the `x` and `y` columns relative to the `patch_size` arguments. The idea is for the user to correctly scaled versions and to use code in `scripts/training/lightning_model.py:BaseTrainer.load_data` to set up the data loader with the logic you need for your data, and then pass a version of that to `celltransformer.data.CenterMaskSampler`
	- for more information on the data and dataloader, see the [data + dataloader page](data.md)
  * alternatively just change the `data.patch_size` config value in `hydra` (see `scripts/config/data/`); as long as the desired patch size and spatial units in the dataframe are correctly scaled, then it will work
4. add `wandb` project if desired to top level config in `config`
5. copy boilerplate for initiating training from `train_aibs_mouse.py` or `train_zhuang.py` (ie code in `main` that); make sure to specify correct config file in the `@hydra.main` decorator
	- for more information on this see the [`hydra` docs](https://hydra.cc/docs/intro/).

## Core code components and usage

### Config management with hydra
The main interface to the training code we wrote is through `hydra` (https://hydra.cc/), which is a configuration framework that uses yaml files to orchestrate and organize complex workflows. Please see the [`hydra`](https://hydra.cc/docs/intro/) documentation for more information.

The pipeline controls the basic training operations through these yaml files and Pytorch Lightning.

For example, `scripts/config/model/base.yaml` controls the parameters of the transformer itself, for example:

```
_target_: celltransformer.model.CellTransformer
encoder_embedding_dim: 384
decoder_embedding_dim: 384

encoder_num_heads: 8
decoder_num_heads: 8
attn_pool_heads: 8

encoder_depth: 4
decoder_depth: 4

cell_cardinality: 384
eps: 1e-9

n_genes: 500
xformer_dropout: 0.0
bias: True
zero_attn: True
```

We can use hydra to directly instantiate this model (which we specify using the `_target_` attribute) by specifiying the object class, here `celltransformer.model.CellTransformer`. What this looks like in context is in following snippet:

```
cfg_path = 'config.yaml'
cfg = OmegaConf.load(cfg_path) # same as above snippet
model = hydra.utils.instantiate(cfg.model)

# model will have 500 gene output decoder depth of 4, etc. and will be an instance of class `celltransformer.model.CellTransformer`
```

### Composition of config files is controlled at top-level using another config

An example of this composition at high level is the `scripts/config/example.yaml` file, which contains the settings used to train on the Allen Institute for Brain Science MERFISH data in the [Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas). Note that "mouse1.yaml" refers to a file inside the `config/data` directory. Correspondingly there is a `config/model/base.yaml` file that is specified by the below config, which is found one-level-up (ie in the `config` directory). 

```
defaults:
  - _self_
  - data: mouse1.yaml
  - model: base.yaml
  - optimization: base.yaml

checkpoint_dir: 
wandb_project: 
model_checkpoint: 
wandb_code_dir: 
```

Where you can see we can group and order config components and define several high level attributes such as the checkpoint directory. You may like, however, to change these. For example including `wandb_project` will assume you can `wandb.login()` (see the wandb website for information on wandb and how to get a free account) and set this as the project. 

The `wandb_code_dir` argument will be used later to log the specific code used. 

For some files, a field may read: `???` indicating that field must be filled or `hydra` will error. 

Overall, fields in config files are accessible as `.[attribute]` in the DictConfig (from Omegaconf) object for example `config.model.n_genes`.

Keep in mind that for dataset paths, all of them ought to be hardcoded in. Therefore, for datapaths in `config/data` these paths should be considered placeholders for you to fill in. I left in paths to explicitly indicate filepaths to the Zhuang and AIBS MERFISH data hosted on [https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html).

### Training on the AIBS MERFISH data

The entrypoint to the training used for the Allen Institute for Brain Science MERFISH dataset (mouse 6388550) is in `scripts/train_aibs_mouse.py`, which uses `scripts/config/aibs1.yaml`. To run the code (assuming the package has been installed):

1. download the data (use `scripts/download_aibs.sh`)
2. edit `config/data/mouse1.yaml`, specifically:
```
adata_path: './abc_dataset/C57BL6J-6388550-log2.h5ad'
metadata_path: './abc_dataset/cell_metadata_with_cluster_annotation.csv'
```
3. change whatever combination of checkpoint and `wandb` settings in `scripts/config/aibs1.yaml`
4. run the trainer script (`scripts/training/train_aibs_mouse.py`)

```
chmod +x scripts/download_aibs.sh
./scripts/download_aibs.sh
python scripts/training/train_aibs_mouse.py
```

If this is useful to you, please consider citing our preprint:

```
@article {Lee2024.05.05.592608,
	author = {Lee, Alex J. and Dubuc, Alma and Kunst, Michael and Yao, Shenqin and Lusk, Nicholas and Ng, Lydia and Zeng, Hongkui and Tasic, Bosiljka and Abbasi-Asl, Reza},
	title = {Data-driven fine-grained region discovery in the mouse brain with transformers},
	elocation-id = {2024.05.05.592608},
	year = {2024},
	doi = {10.1101/2024.05.05.592608},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Technologies such as spatial transcriptomics offer unique opportunities to define the spatial organization of the mouse brain. We developed an unsupervised training scheme and novel transformer-based deep learning architecture to detect spatial domains across the whole mouse brain using spatial transcriptomics data. Our model learns local representations of molecular and cellular statistical patterns which can be clustered to identify spatial domains within the brain from coarse to fine-grained. Discovered domains are spatially regular, even with several hundreds of spatial clusters. They are also consistent with existing anatomical ontologies such as the Allen Mouse Brain Common Coordinate Framework version 3 (CCFv3) and can be visually interpreted at the cell type or transcript level. We demonstrate our method can be used to identify previously uncatalogued subregions, such as in the midbrain, where we uncover gradients of inhibitory neuron complexity and abundance. Notably, these subregions cannot be discovered using other methods. We apply our method to a separate multi-animal whole-brain spatial transcriptomic dataset and show that our method can also robustly integrate spatial domains across animals.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/06/13/2024.05.05.592608},
	eprint = {https://www.biorxiv.org/content/early/2024/06/13/2024.05.05.592608.full.pdf},
	journal = {bioRxiv}
}
```

## Acknowledgments
This documentation was copied from Patrick Kidger's `jaxtyping` [docs](https://github.com/patrick-kidger/jaxtyping).