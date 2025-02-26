import argparse
import logging
from pathlib import Path
import time
import warnings

import pandas as pd
import numpy as np
from sklearn import preprocessing
#import tqdm
from rich.progress import track, Progress

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s: [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file. We expect a CSV.", required=True)
    parser.add_argument('--celltype_level', type=str, help='The level of cell type to extract.', required=True)
    parser.add_argument('--sort_metadata', action='store_true', help='Sort metadata by region.', required=False)

    workflow_type = parser.add_mutually_exclusive_group(required=True)

    workflow_type.add_argument('--labels_path', type=str, nargs='+', help='Path(s) to labels file(s). We expect `npy` file(s).', required=False)
    #workflow_type.add_argument('--labels_path', type=str, help='Path to labels file. We expect a `npy` file.', required=False)
    workflow_type.add_argument('--colname', type=str, help='Column name in metadata file that contains the labels.', required=False)
    workflow_type.add_argument('--section_colname', type=str, help='Column name in metadata file that contains the section labels.', required=False, default='section')   

    parser.add_argument('--output_fname', type=str, help='Filename to save the matrix.', required=False)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file.', required=False, default=False)

    return parser.parse_args()

def extract_celltype_labels(df: pd.DataFrame, 
                          matrix: np.ndarray, 
                          celltype_target_col: str, 
                          region_colname: str,
                          #section_colname: str = 'section',
                          ):
    # if celltype_target_col not in df.columns:
    #     raise ValueError(f'Target column: {celltype_target_col} not found in dataframe.')
    
    # if section_colname not in df.columns:
    #     raise ValueError(f'Section column: {section_colname} not found in dataframe.')
    
    # if region_colname not in df.columns:
    #     raise ValueError(f'Region colname: {region_colname} not found in dataframe.')
    
    grouped = df.groupby(region_colname)
    total = 0
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(grouped))
        for region_index, groupby in grouped:
            value_counts = groupby[celltype_target_col].value_counts()
            matrix[region_index, value_counts.index] = value_counts.values
            num_cells = value_counts.values.sum()
            total += num_cells
            progress.update(task, advance=1, description=f'[cyan] Counted: {num_cells} cells in region {region_index}; {total} total cells.')

    return matrix


from scipy import optimize
def compare_count_matrices(m_a: np.ndarray, 
                           m_b: np.ndarray, 
                           log: bool = True,
                           axis: int = 1):
    
    if m_a.shape[1] != m_b.shape[1]:
        raise ValueError(f"Matrix number of columns must be same: {m_a.shape} != {m_b.shape}")
    
    k = m_a.shape[axis]
        
    if log:
        m_a = np.log1p(m_a)
        m_b = np.log1p(m_b)

    corr_mat = np.corrcoef(m_a, m_b)[:k, k:]
    row_inds, col_inds = optimize.linear_sum_assignment(corr_mat, maximize=True)

    return corr_mat, row_inds, col_inds

    

def main():
    args = parse_args()

    try:
        # supresss pandas mixed datatypes warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
            metadata = pd.read_csv(args.metadata_path)

    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    logging.info(f"Loaded metadata from {args.metadata_path}.")
    logging.info('Length: %s', len(metadata))
    
    if args.celltype_level not in metadata.columns:
        raise ValueError(f"Column name not found in metadata: {args.celltype_level}")
    
    if args.section_colname not in metadata.columns:
        raise ValueError(f"Column name not found in metadata: {args.section_colname}")
    
    if args.sort_metadata:
        logging.info('Sorting metadata by region.')
        reordered_groupby = []
        for region_index, groupby in metadata.groupby(args.section_colname):
            reordered_groupby.append(groupby)
        metadata = pd.concat(reordered_groupby)
    
    elif args.colname:
        if args.colname not in metadata.columns:
            raise ValueError(f"Column name not found in metadata: {args.colname}")
        
        metadata['region'] = preprocessing.LabelEncoder().fit_transform(metadata[args.colname])

    l_enc_celltype = preprocessing.LabelEncoder()
    l_enc_celltype.fit(sorted(metadata[args.celltype_level].unique()))

    metadata['celltype_tocount'] = l_enc_celltype.transform(metadata[args.celltype_level])


    if len(args.labels_path) == 1:
        args.labels_path = args.labels_path[0]
        try:
            labels = np.load(args.labels_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
        
        metadata['region'] = preprocessing.LabelEncoder().fit_transform(labels) # these will be in order anyway, but just in case
            
        # we need to fit the encoders so that it's easy to then later index into the `celltype_composition_matrix`
        # otherwise I suppose we could have just kept two dictionaries or something 

        #l_enc_region = preprocessing.LabelEncoder()
        #l_enc_region.fit(sorted(metadata['region'].unique()))

        #metadata['region'] = l_enc_region.transform(metadata['region'])
        logging.info('Finished preprocessing spatial cluster and cell type data.')

        num_rows = metadata['region'].nunique()
        num_cols = metadata['celltype_tocount'].nunique()

        logging.info(f'Forming matrix with: {num_rows} x {num_cols} size.')

        celltype_composition_matrix = np.zeros((num_rows, num_cols), dtype=np.uint32)

        # import sys
        # sys.exit(1)

        logging.info('Extracting cell type composition matrix!')
        celltype_composition_matrix = extract_celltype_labels(metadata, celltype_composition_matrix, 
                                                            'celltype_tocount', 'region',
        )
        
        np.save(args.output_fname, celltype_composition_matrix)
        logging.info(f'Saved cell type composition matrix to: {args.output_fname}')
    else:
        for labels_path in args.labels_path:
            tnow = time.time()
            output_name = labels_path.replace('.npy', f'_celltype_composition-{args.celltype_level}.npy')
            if Path(output_name).exists():
                logging.info(f'Skipping: {output_name} already exists.')
                if not args.overwrite:
                    continue
                

            try:
                labels = np.load(labels_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
            metadata['region'] = preprocessing.LabelEncoder().fit_transform(labels)

            logging.info('Finished preprocessing spatial cluster and cell type data for file: %s', labels_path)

            num_rows = metadata['region'].nunique()
            num_cols = metadata['celltype_tocount'].nunique()

            logging.info(f'Forming matrix with: {num_rows} x {num_cols} size.')

            celltype_composition_matrix = np.zeros((num_rows, num_cols), dtype=np.uint32)

            logging.info('Extracting cell type composition matrix')
            celltype_composition_matrix = extract_celltype_labels(metadata, celltype_composition_matrix, 
                                                                'celltype_tocount', 'region',
                                                )

            np.save(output_name, celltype_composition_matrix)
            t_complete = time.time() - tnow
            logging.info(f'Saved cell type composition matrix to: {output_name} in {t_complete:.2f} seconds.')

    logging.info('All done!')

if __name__ == '__main__':
    main()