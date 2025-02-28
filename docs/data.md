## Dataloader information
We use a packed sequence format, so the returns from this function are not in general `(batch_size, length, features)` tensor but a `(length, features)` matrix. 

Here is the docstring for the loader:

::: celltransformer.data.CenterMaskSampler.__init__
	handler: python
	options:
	  show_source: false
	  show_root_heading: false
	  merge_init_into_class: true
	  

The required data components are an `anndata` object with keys corresponding to columns in a dataframe of metadata, `metadata`. 

The dataframe **must** have:

* x: spatial coordinates **[default: x]**
* y: spatial coordinates **[default: y]**
* one column (`cell_id_colname`) which refers to the keys, these will be used directly to subset into the `anndata` object **[default: cell_label]**
* one column (`cell_type_colname`) which is a class encoded integer corresponding to the single cell classes from the non-spatial scRNA-seq data clustering **[default: cell_type]**

The units of `patch_size` are not scaled internally, so they must "match" the units of `x` and `y`. 

What if you don't have a dataframe with those columns (and therefore is incorrectly formatted for `celltransformer.data.CenterMaskSampler`)? Your options are:

1. just create a new version of the dataframe with the correct column names and metadata 
2. use the `scripts/training/lightning_model.py:BaseTrainer.load_data` function to preprocess your data in the format that will satisfy the conditions (see `scripts/training/train_aibs_mouse.py:load_data`) as an example. The motivations for this are covered in the TLDR in the main page, but we will also discuss it here. 

### Expectations for inputs into `celltransformer.data.CenterMaskSampler`, using `scripts/training/train_aibs_mouse.py:load_data` as an example

1. we need columns `x` and `y` which match the units of `patch_size`
	- e.g. if your patch size desired size is 10um, and your spatial dimensions are provided in nm, you would want to rescale either the patch size or the spatial dimensions. I elected to primarily rescale the spatial dimensions (for my own sanity) all to um, and so use the `load_data` function do do so.
2. we also need maybe to remap `cell_type_colname` to give 

Therefore, let's annotate the test code from `train_aibs_mouse.py:load_data`:

```
# make sure to encode as str because for whatever reason
# in the AIBS data, the anndata row ID's are string instead of int
metadata["cell_label"] = metadata["cell_label"].astype(str) 
metadata["x"] = metadata["x_reconstructed"] * 100 # initially x_reconstructed in wrong units
metadata["y"] = metadata["y_reconstructed"] * 100

metadata = metadata[ # throw away nonessential columns
	[
		config.data.celltype_colname,
		"cell_label", 
		"x",
		"y",
		"brain_section_label",
	]
]

# label_to_cls is just a thin wrapper over 
# `sklearn.preprocessing.LabelEncoder`
metadata["cell_type"] = self.label_to_cls(
	metadata[config.data.celltype_colname]
)
# make sure to integer encoder the classes 
metadata["cell_type"] = metadata["cell_type"].astype(int)

metadata = metadata.reset_index(drop=True)

# in the AIBS dataset some cells for which gene 
# expression was measured do not have metadata associated
# so let's throw those out
adata = adata[metadata["cell_label"]]


```

<br>

## Outputs of `__getitem__` and `collate`

These are the steps we implement so far:

1. extract spatial neighbors (within some radius) for the cell of interest
2. extract expression (from the `anndata` object) and the class encoding (`cell_type`) for each cell from the metadata dataframe you should have passed to `CenterMaskSampler`
3. partition them into two sets, which are simply lists in a `namedtuple`, see `celltransformer.data.NeighborhoodMetadata`

Note that we then need to stack these together across batches and create attention matrix masks, which is done in `celltransformer.data.loader_pandas.collate`. We create three attention mask matrices (we use a binary adjacency matrix):

| Syntax      | Description |
| ----------- | ----------- |
| `encoder_mask`   | allow the neighborhoods to only attend to each other       |
| `pooling_mask`   | pool into a single query token        |
| `decoder_mask`   | allow query token and decoding cell tokens to attend to each other				| 

See the function documentation and the next section [model](model.md) for more information. 

My first-shot implementation of this workflow (in `pandas`) was ~1.3X faster than my next attempt, in `polars`. I welcome any feedback on why my `polars` code may have been suboptimal!
