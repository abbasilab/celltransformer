echo "Starting download of ABC MERFISH data:"
mkdir -p abc_dataset
wget -P abc_dataset/  https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad 
echo "Started downloading metadata:"
wget -P abc_dataset/ https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/MERFISH-C57BL6J-638850/20231215/views/cell_metadata_with_cluster_annotation.csv
echo "Finished downloading ABC MERFISH data."
