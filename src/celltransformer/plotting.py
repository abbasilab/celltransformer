"""Shared spatial plotting utilities for celltransformer.

make_section_plot
    Scatter-plot a single tissue section from a DataFrame of per-cell spatial
    coordinates and a colour column.  Used by ``celltransformer-cluster`` and
    available for any downstream notebook or script that needs a quick
    section-level spatial visualisation.
"""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


def make_section_plot(
    section_name: str,
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    color_col: str = "datadrivencolor",
    section_col: str = "section",
    s: float = 2.5,
    xbounds: Optional[Tuple[float, float]] = None,
    ybounds: Optional[Tuple[float, float]] = None,
    axis: bool = False,
    dpi: int = 300,
    title: Union[bool, str] = False,
    save: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a scatter plot for a specific tissue section.

    Parameters
    ----------
    section_name : str
        Value in *section_col* that selects the rows to plot.
    df : pd.DataFrame
        DataFrame containing spatial coordinates and colours.
    x_col, y_col : str
        Column names for spatial coordinates.
    color_col : str
        Column whose values are passed as ``c`` to ``ax.scatter``.
    section_col : str
        Column used to filter *df* to the requested section.
    s : float
        Marker size.
    xbounds, ybounds : tuple of (min, max), optional
        Axis limits.  When *ybounds* is ``None`` the y-axis is inverted
        (MERFISH convention: y increases downward).
    axis : bool
        Show axis ticks and frame.
    dpi : int
        Figure resolution.
    title : bool or str
        ``False`` → no title; ``True`` → use *section_name*; ``str`` → literal title.
    save : str, optional
        If given, save the figure to this path (tight bbox, transparent bg).
        Otherwise call ``plt.show()``.

    Returns
    -------
    fig, ax
    """
    sub = df[df[section_col] == section_name]

    if len(sub) == 0:
        raise ValueError(f"No data found for section '{section_name}'")

    fig, ax = plt.subplots(1, figsize=(6.4, 4.8), dpi=dpi)

    ax.scatter(
        x=sub[x_col],
        y=sub[y_col],
        s=s,
        c=sub[color_col],
        edgecolors="black",
        linewidths=0.05,
        zorder=10,
        alpha=0.9,
    )

    if xbounds is not None:
        ax.set_xlim(*xbounds)

    if ybounds is not None:
        ax.set_ylim(*ybounds)
    else:
        ax.invert_yaxis()

    if not axis:
        ax.axis("off")

    if title:
        ax.set_title(section_name if isinstance(title, bool) else title)

    if save:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.savefig(save, dpi=dpi, bbox_inches="tight", transparent=True)
    else:
        plt.show()

    return fig, ax
