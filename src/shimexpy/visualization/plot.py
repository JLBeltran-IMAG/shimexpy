"""
Visualization utilities for displaying and comparing contrast images.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_contrast(contrast, title=None, cmap='gray', figsize=(10, 8)):
    """
    Plot a single contrast image with a colorbar.

    Parameters
    ----------
    contrast : np.ndarray or xr.DataArray
        The contrast image to display.
    title : str, optional
        Title for the plot. Default is None.
    cmap : str, optional
        Matplotlib colormap name. Default is 'gray'.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 8).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(contrast, xr.DataArray):
        im = ax.imshow(contrast.values, cmap=cmap)
    else:
        im = ax.imshow(contrast, cmap=cmap)
    
    if title:
        ax.set_title(title)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig, ax


def plot_multiple_contrasts(contrasts, titles=None, cmap='gray', figsize=(18, 6)):
    """
    Plot multiple contrast images side by side.

    Parameters
    ----------
    contrasts : list
        List of contrast arrays to display.
    titles : list, optional
        List of titles for each contrast image. Default is None.
    cmap : str, optional
        Matplotlib colormap name. Default is 'gray'.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (18, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : list of matplotlib.axes.Axes
        List of the matplotlib Axes objects.
    """
    n = len(contrasts)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, contrast in enumerate(contrasts):
        if isinstance(contrast, xr.DataArray):
            im = axes[i].imshow(contrast.values, cmap=cmap)
        else:
            im = axes[i].imshow(contrast, cmap=cmap)
            
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
            
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    return fig, axes


def compare_before_after(before, after, titles=None, cmap='gray', figsize=(18, 6)):
    """
    Compare two images (e.g., before and after processing) side by side
    and show their difference.

    Parameters
    ----------
    before : np.ndarray or xr.DataArray
        The "before" image.
    after : np.ndarray or xr.DataArray
        The "after" image.
    titles : list, optional
        List of titles [before_title, after_title, diff_title]. Default is None.
    cmap : str, optional
        Matplotlib colormap name. Default is 'gray'.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (18, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : list of matplotlib.axes.Axes
        List of the matplotlib Axes objects.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert to numpy arrays if they are xarray DataArrays
    before_data = before.values if isinstance(before, xr.DataArray) else before
    after_data = after.values if isinstance(after, xr.DataArray) else after
    
    # Compute difference
    diff = after_data - before_data
    
    # Plot images
    im1 = axes[0].imshow(before_data, cmap=cmap)
    im2 = axes[1].imshow(after_data, cmap=cmap)
    im3 = axes[2].imshow(diff, cmap=cmap)
    
    # Set titles
    default_titles = ['Before', 'After', 'Difference']
    if titles is None:
        titles = default_titles
    elif len(titles) < 3:
        titles = titles + default_titles[len(titles):]
    
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    return fig, axes
