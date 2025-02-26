import argparse
import logging

import torch
import numpy as np
import cuml
import pandas as pd
import anndata as ad
import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s [%(filename)s:%(lineno)d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True, help='Path to the metadata CSV that holds the spatial coordinates.')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to the embedding file.')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--gaussian_kernel', type=int, default=100, help='Gaussian kernel size for smoothing (NOTE: UNITS ARE ARBITRARY).')
    parser.add_argument('--spatial_x_colname', type=str, default='x')
    parser.add_argument('--spatial_y_colname', type=str, default='y')
    parser.add_argument('--section_colname', type=str, default='section')
    parser.add_argument('--n_neighbors', type=int, default=500, help='Number of neighbors to use for the approximate kNN+Gaussian smoothing operation.')

    return parser.parse_args()

def gaussian(x, sigma):
    return torch.exp(-x ** 2 / (2 * sigma ** 2))

def main():
    args = parse_args()

    metadata = pd.read_csv(args.metadata_path)
    metadata = metadata.reset_index(drop=True)
    logging.info('Loaded metadata.')

    embedding = torch.load(args.embedding_path)
    logging.info('Loaded embedding.')

    assert len(metadata) == len(embedding), "Metadata and embedding have different lengths"

    smoothed_embeds = []

    count = 0
    tot_count = metadata[args.section_colname].nunique()
    for section, section_groupby in metadata.groupby(args.section_colname):
        logging.info(f'Started smoothing for: {section} [{count+1}/{tot_count}]')

        X = section_groupby[[args.spatial_x_colname, args.spatial_y_colname]].values

        logging.info('Fitting KNN...')
        knn = cuml.NearestNeighbors(n_neighbors=args.n_neighbors)
        knn.fit(X)
        dists, indices = knn.kneighbors(X)

        logging.info('Smoothing...')
        weights = gaussian(torch.from_numpy(dists), args.gaussian_kernel)
        normalized_wts = weights / weights.sum(axis=1)[:, None]
        logging.info('Computed weights.')

        indices_X = torch.from_numpy(indices)

        logging.info('Computing new smoothed features...')
        embeds_neighbors = embedding[indices_X]
        weighted = embeds_neighbors * normalized_wts[:, :, None]
        embeds_weighted = weighted.sum(axis=1)
        logging.info('Smoothing complete!')

        smoothed_embeds.append(embeds_weighted)
    
    smoothed_embeds = torch.cat(smoothed_embeds)
    logging.info('Saving smoothed embeddings: ' + str(args.output_path))
    torch.save(smoothed_embeds, args.output_path)
        
if __name__ == "__main__":
    main()





