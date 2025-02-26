import argparse
import random
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

"""
Script to 
    

"""

def plot_2d(df_section, 
            s=2.5,
            xbounds=(9.01, 2.0),
            ybounds=(0.2, 10.7),
            axis=False,
            dpi=100,
            title=False,
            save=None,) -> None:
    
    fig, axs = plt.subplots(1)#, figsize=(6.4, 4.8))
    
    g = plt.scatter(x=df_section.spatial_x, 
                    y=df_section.spatial_y, 
                    s=s,
                    c=df_section.spatial_cluster_color,
                    edgecolors='black',
                    linewidths=0.05,
                    zorder=10,
                    alpha=0.9)
    
    if xbounds is not None:
        axs.set_xlim(*xbounds)
    if ybounds is not None:
        axs.set_ylim(*ybounds)
    # if xbounds is None and ybounds is None:
    #     axs.invert_yaxis()
        
    if axis is False:
        axs.axis('off')
        
    plt.gcf().set_dpi(dpi)
    
    if title:
        plt.title(title)
        
    if save:
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.savefig(save, dpi=dpi, bbox_inches='tight', transparent=True)
    else:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file. We expect a CSV.", required=True)
    parser.add_argument('--labels_path', type=str, help='Path to labels file. We expect an `npy` file.', required=True)
    
    opts = parser.add_argument_group('Options and flags')
    opts.add_argument('--title', type=bool, help='Put section label titles on the plot.', required=False, default=True)
    opts.add_argument('--seed', type=int, default=42, help='Random seed')
    opts.add_argument('--num_shuffle', default=1, type=int, help='Number of times to shuffle the labels.')
    opts.add_argument('--dpi', type=int, default=100, help='DPI for the plot.')
    opts.add_argument('--s', type=float, default=2.5, help='Size of the points in the plot.')
    opts.add_argument('--xbounds', type=tuple, default=(9.01, 2.0), help='X-axis bounds for the plot.')
    opts.add_argument('--ybounds', type=tuple, default=(0.2, 10.7), help='Y-axis bounds for the plot.')
    opts.add_argument('--axis', type=bool, default=False, help='Show axis on the plot.')

    opts.add_argument('--spatial_x', type=str, default='spatial_x', help='Column name for the x-axis.')
    opts.add_argument('--spatial_y', type=str, default='spatial_y', help='Column name for the y-axis.')

    parser.add_argument('--output_dirname', type=str, help='Path to save embeddings.', required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    try:
        metadata = pd.read_csv(args.metadata_path)
    except:
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    try:
        labels = np.load(args.labels_path)
    except:
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
    
    if len(metadata) != len(labels):
        raise ValueError(f"Metadata and labels have different lengths: {len(metadata)} and {len(labels)}")
    
    try:
        seed = int(args.seed)
    except:
        raise ValueError(f"Seed must be an integer: {args.seed}")

    n_clust = np.unique(labels).shape[0]

    colormap = sns.color_palette(cc.glasbey, n_colors=n_clust)

    try:
        num_shuffle = int(args.num_shuffle)
    except:
        raise ValueError(f"Number of shuffles must be an integer: {args.num_shuffle}.")
    
    try:
        num_shuffle = int(args.num_shuffle)
        if num_shuffle < 0:
            raise ValueError(f"Number of shuffles must be a positive integer: {num_shuffle}.")
    except:
        raise ValueError(f"Number of shuffles must be an integer greater than 0: {args.num_shuffle}.")

    for _ in range(num_shuffle):
        random.Random(seed).shuffle(labels)
    
    colormap_hex = {k: v for k, v in zip(range(len(colormap)), colormap.as_hex())}

    label_color = [colormap_hex[l] for l in labels]
    metadata['spatial_cluster_color'] = label_color

    output_path = pathlib.Path(args.output_dirname)
    if output_path.exists() is False:
        output_path.mkdir(parents=True, exist_ok=True)

    spatial_x_col = args.spatial_x
    if spatial_x_col != 'spatial_x':
        metadata['spatial_x'] = metadata[spatial_x_col]

    spatial_y_col = args.spatial_y
    if spatial_y_col != 'spatial_y':
        metadata['spatial_y'] = metadata[spatial_y_col]

    for section, groupby in metadata.groupby('brain_section_label'):
        plot_2d(groupby, 
                s=args.s,
                xbounds=args.xbounds,
                ybounds=args.ybounds,
                axis=args.axis,
                dpi=args.dpi,
                title=args.title,
                save=f"{output_path}/{section}.png")


if __name__ == '__main__':
    main()

