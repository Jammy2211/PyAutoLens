import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autoarray.plot.plots.array import plot_array
from autolens.plot.plot_utils import (
    _to_lines,
    _save_subplot,
    _critical_curves_from,
)


def _plot_yx(y, x, ax, title, xlabel="", ylabel=""):
    """Scatter plot of y vs x into an axes."""
    ax.scatter(x, y, s=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                       colormap="jet", use_log10=False):
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    if not tracer.planes[plane_index].has(cls=aa.Pixelization):
        zoom = aa.Zoom2D(mask=fit.dataset.real_space_mask)
        grid = aa.Grid2D.from_extent(
            extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
        )
        traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
        plane_galaxies = ag.Galaxies(galaxies=tracer.planes[plane_index])
        image = plane_galaxies.image_2d_from(grid=traced_grids[plane_index])
        plot_array(
            array=image, ax=ax,
            title=f"Source Plane {plane_index}",
            colormap=colormap, use_log10=use_log10,
        )
    else:
        if ax is not None:
            ax.axis("off")
            ax.set_title(f"Source Reconstruction (plane {plane_index})")


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """12-panel subplot for an interferometer fit."""
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    final_plane_index = len(fit.tracer.planes) - 1

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes_flat = list(axes.flatten())

    # Panel 0: amplitudes vs UV distances
    _plot_yx(
        y=np.real(fit.residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[0],
        title="Amplitudes vs UV-Distance",
        xlabel=r"k$\lambda$",
    )

    plot_array(array=fit.dirty_image, ax=axes_flat[1], title="Dirty Image",
               colormap=colormap)
    plot_array(array=fit.dirty_signal_to_noise_map, ax=axes_flat[2],
               title="Dirty Signal-To-Noise Map", colormap=colormap)
    plot_array(array=fit.dirty_model_image, ax=axes_flat[3], title="Dirty Model Image",
               colormap=colormap)

    # Panel 4: source image
    _plot_source_plane(fit, axes_flat[4], final_plane_index, colormap=colormap)

    # Normalized residual vs UV distances (real)
    _plot_yx(
        y=np.real(fit.normalized_residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[5],
        title="Norm Residual vs UV-Distance (real)",
        ylabel=r"$\sigma$",
        xlabel=r"k$\lambda$",
    )

    # Normalized residual vs UV distances (imag)
    _plot_yx(
        y=np.imag(fit.normalized_residual_map),
        x=fit.dataset.uv_distances / 10 ** 3.0,
        ax=axes_flat[6],
        title="Norm Residual vs UV-Distance (imag)",
        ylabel=r"$\sigma$",
        xlabel=r"k$\lambda$",
    )

    # Panel 7: source plane zoomed
    _plot_source_plane(fit, axes_flat[7], final_plane_index, colormap=colormap)

    plot_array(array=fit.dirty_normalized_residual_map, ax=axes_flat[8],
               title="Dirty Normalized Residual Map", colormap=colormap)

    # Panel 9: clipped to [-1, 1]
    from autolens.imaging.plot.fit_imaging_plots import _plot_with_vmin_vmax
    _plot_with_vmin_vmax(fit.dirty_normalized_residual_map, axes_flat[9],
                         r"Normalized Residual Map $1\sigma$", colormap,
                         vmin=-1.0, vmax=1.0)

    plot_array(array=fit.dirty_chi_squared_map, ax=axes_flat[10],
               title="Dirty Chi-Squared Map", colormap=colormap)

    # Panel 11: source plane not zoomed
    _plot_source_plane(fit, axes_flat[11], final_plane_index,
                       zoom_to_brightest=False, colormap=colormap)

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit", output_format)


def subplot_fit_real_space(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """Real-space subplot: image + source plane (or inversion panels)."""
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    final_plane_index = len(fit.tracer.planes) - 1

    if fit.inversion is None:
        # No inversion: image + source plane image
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes_flat = list(axes.flatten())

        zoom = aa.Zoom2D(mask=fit.dataset.real_space_mask)
        grid = aa.Grid2D.from_extent(
            extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
        )
        traced_grids = tracer.traced_grid_2d_list_from(grid=grid)

        image = tracer.image_2d_from(grid=grid)
        plot_array(array=image, ax=axes_flat[0], title="Image", colormap=colormap)

        source_galaxies = ag.Galaxies(galaxies=tracer.planes[final_plane_index])
        source_image = source_galaxies.image_2d_from(
            grid=traced_grids[final_plane_index]
        )
        plot_array(array=source_image, ax=axes_flat[1], title="Source Plane Image",
                   colormap=colormap)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes_flat = list(axes.flatten())
        # Inversion: show blank placeholder panels
        for _ax in axes_flat:
            _ax.axis("off")
        axes_flat[0].set_title("Reconstructed Data")
        axes_flat[1].set_title("Source Reconstruction")

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit_real_space", output_format)
