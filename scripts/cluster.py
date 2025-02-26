# NOTE: you need cuml for this to work, which is not installed by default in this requirements.txt
# write me a script that will argparse one item (embeds_path) and load it using torch.load
# then use cuml to KMeans the embeddings into 10 clusters

import argparse
import pathlib
import logging

import cuml
import cupy
import numpy as np
import torch

import joblib

# set up basic logger message
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeds_path", type=str, help="Path to embeddings file.", required=True)
    parser.add_argument("--n_clust", type=int, help="Number of clusters", default=100)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--maxiter', type=int, default=1500, help='Maximum number of iterations')
    parser.add_argument('--n_init', type=int, default=3, help='Number of initializations')
    parser.add_argument('--oversampling_factor', type=int, default=3, help='Oversampling factor')
    parser.add_argument('--tol', type=float, default=1e-10, help='Tolerance')
    parser.add_argument('--output_fstem', type=str, default='kmeans-cluster_labels.npy', help='Output filename')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cupy.random.seed(args.seed)

    embeds_path = pathlib.Path(args.embeds_path)

    logging.info(f"Loading embeddings from {embeds_path}.")

    if embeds_path.exists() is False:
        raise FileNotFoundError(f'Embeddings file not found: {args.embeds_path}')

    if args.output_fname is None:
        raise ValueError("Output filename must be specified.")
    
    output_fname = pathlib.Path(args.output_fname)

    if output_fname.exists():
        logging.warning(f"Output file {args.output_fname} already exists. Overwriting.")

    if output_fname.parent.exists() is False:
        logging.info(f"Creating output directory {output_fname.parent}.")
        output_fname.parent.mkdir(parents=True, exist_ok=True)
   
    if embeds_path.suffix in ('.pth', 'pt'):
        embeds = torch.load(embeds_path)
        embeds = embeds.numpy()
    elif embeds_path.suffix in ('.npy',):
        embeds = np.load(embeds_path)

    logging.info(f"Loaded embeddings of shape {embeds.shape}.")
    logging.info(f"Clustering into {args.n_clust} clusters.")
    
    kmeans = cuml.KMeans(n_clusters=args.n_clust, 
                         max_iter=args.maxiter, 
                         n_init=args.n_init, 
                         oversampling_factor=args.oversampling_factor,
                         tol=args.tol,
                         random_state=args.seed)
    kmeans.fit(embeds)
    logging.info("Finished clustering.")
    labels = kmeans.predict(embeds)

    logging.info(f"Saving cluster labels to {output_fname}_clusterlabels.npy.")

    np.save(f'{output_fname}_clusterlabels.npy', labels)

    joblib.dump(kmeans, f'{output_fname}_kmeans_k{args.n_clust}.joblib')

if __name__ == "__main__":
    main()