import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag

from autoarray.plot.plots.array import plot_array, _zoom_array_2d
from autoarray.plot.plots.utils import save_figure
from autolens.plot.plot_utils import (
    _to_lines,
    _critical_curves_from,
    _caustics_from,
)


def _get_source_vmax(fit):
    """Return vmax based on source-plane model images, or None."""
    try:
        return float(np.max([mi.array for mi in fit.model_images_of_planes_list[1:]]))
    except (ValueError, IndexError):
        return None


def _plot_source_plane(fit, ax, plane_index, zoom_to_brightest=True,
                       colormap="jet", use_log10=False):
    """Plot source plane image or inversion reconstruction into *ax*."""
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    if not tracer.planes[plane_index].has(cls=aa.Pixelization):
        zoom = aa.Zoom2D(mask=fit.mask)
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
        # Inversion path: in subplot context show a blank panel.
        if ax is not None:
            ax.axis("off")
            ax.set_title(f"Source Reconstruction (plane {plane_index})")


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    plane_index: Optional[int] = None,
):
    """12-panel subplot of the imaging fit.

    For single-plane tracers delegates to :func:`subplot_fit_x1_plane`.
    """
    if len(fit.tracer.planes) == 1:
        return subplot_fit_x1_plane(fit, output_path=output_path,
                                    output_format=output_format, colormap=colormap)

    plane_index_tag = "" if plane_index is None else f"_{plane_index}"
    final_plane_index = (
        len(fit.tracer.planes) - 1 if plane_index is None else plane_index
    )

    source_vmax = _get_source_vmax(fit)

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes_flat = list(axes.flatten())

    plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap)

    # Data at source scale
    plot_array(array=fit.data, ax=axes_flat[1], title="Data (Source Scale)",
               colormap=colormap, vmax=source_vmax)

    plot_array(array=fit.signal_to_noise_map, ax=axes_flat[2],
               title="Signal-To-Noise Map", colormap=colormap)
    plot_array(array=fit.model_data, ax=axes_flat[3], title="Model Image",
               colormap=colormap)

    # Lens model image
    try:
        lens_model_img = fit.model_images_of_planes_list[0]
    except (IndexError, AttributeError):
        lens_model_img = None
    if lens_model_img is not None:
        plot_array(array=lens_model_img, ax=axes_flat[4],
                   title="Lens Light Model Image", colormap=colormap)
    else:
        axes_flat[4].axis("off")

    # Subtracted image at source scale
    try:
        subtracted_img = fit.subtracted_images_of_planes_list[final_plane_index]
    except (IndexError, AttributeError):
        subtracted_img = None
    if subtracted_img is not None:
        plot_array(array=subtracted_img, ax=axes_flat[5], title="Lens Light Subtracted",
                   colormap=colormap, vmin=0.0 if source_vmax is not None else None,
                   vmax=source_vmax)
    else:
        axes_flat[5].axis("off")

    # Source model image at source scale
    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
    except (IndexError, AttributeError):
        source_model_img = None
    if source_model_img is not None:
        plot_array(array=source_model_img, ax=axes_flat[6], title="Source Model Image",
                   colormap=colormap, vmax=source_vmax)
    else:
        axes_flat[6].axis("off")

    # Source plane zoomed
    _plot_source_plane(fit, axes_flat[7], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap)

    # Normalized residual map (symmetric)
    norm_resid = fit.normalized_residual_map
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[8], title="Normalized Residual Map",
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max)

    # Normalized residual map clipped to [-1, 1]
    plot_array(array=norm_resid, ax=axes_flat[9],
               title=r"Normalized Residual Map $1\sigma$",
               colormap=colormap, vmin=-1.0, vmax=1.0)

    plot_array(array=fit.chi_squared_map, ax=axes_flat[10],
               title="Chi-Squared Map", colormap=colormap)

    # Source plane not zoomed
    _plot_source_plane(fit, axes_flat[11], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename=f"subplot_fit{plane_index_tag}", format=output_format)


def subplot_fit_x1_plane(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """6-panel subplot for a single-plane tracer fit."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes_flat = list(axes.flatten())

    try:
        vmax = float(np.max(fit.model_images_of_planes_list[0].array))
    except (IndexError, AttributeError, ValueError):
        vmax = None

    plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap, vmax=vmax)

    plot_array(array=fit.signal_to_noise_map, ax=axes_flat[1],
               title="Signal-To-Noise Map", colormap=colormap)

    plot_array(array=fit.model_data, ax=axes_flat[2], title="Model Image",
               colormap=colormap, vmax=vmax)

    norm_resid = fit.normalized_residual_map
    plot_array(array=norm_resid, ax=axes_flat[3], title="Lens Light Subtracted",
               colormap=colormap)

    plot_array(array=norm_resid, ax=axes_flat[4], title="Subtracted Image Zero Minimum",
               colormap=colormap, vmin=0.0)

    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[5], title="Normalized Residual Map",
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_fit_x1_plane", format=output_format)


def subplot_fit_log10(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    plane_index: Optional[int] = None,
):
    """12-panel log10 subplot of the imaging fit."""
    if len(fit.tracer.planes) == 1:
        return subplot_fit_log10_x1_plane(fit, output_path=output_path,
                                          output_format=output_format, colormap=colormap)

    plane_index_tag = "" if plane_index is None else f"_{plane_index}"
    final_plane_index = (
        len(fit.tracer.planes) - 1 if plane_index is None else plane_index
    )

    source_vmax = _get_source_vmax(fit)

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes_flat = list(axes.flatten())

    plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap,
               use_log10=True)

    try:
        plot_array(array=fit.data, ax=axes_flat[1], title="Data (Source Scale)",
                   colormap=colormap, use_log10=True)
    except ValueError:
        axes_flat[1].axis("off")

    try:
        plot_array(array=fit.signal_to_noise_map, ax=axes_flat[2],
                   title="Signal-To-Noise Map", colormap=colormap, use_log10=True)
    except ValueError:
        axes_flat[2].axis("off")

    plot_array(array=fit.model_data, ax=axes_flat[3], title="Model Image",
               colormap=colormap, use_log10=True)

    try:
        lens_model_img = fit.model_images_of_planes_list[0]
        plot_array(array=lens_model_img, ax=axes_flat[4],
                   title="Lens Light Model Image", colormap=colormap, use_log10=True)
    except (IndexError, AttributeError):
        axes_flat[4].axis("off")

    try:
        subtracted_img = fit.subtracted_images_of_planes_list[final_plane_index]
        plot_array(array=subtracted_img, ax=axes_flat[5],
                   title="Lens Light Subtracted", colormap=colormap, use_log10=True)
    except (IndexError, AttributeError):
        axes_flat[5].axis("off")

    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
        plot_array(array=source_model_img, ax=axes_flat[6],
                   title="Source Model Image", colormap=colormap, use_log10=True)
    except (IndexError, AttributeError):
        axes_flat[6].axis("off")

    _plot_source_plane(fit, axes_flat[7], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap, use_log10=True)

    norm_resid = fit.normalized_residual_map
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[8], title="Normalized Residual Map",
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max)

    plot_array(array=norm_resid, ax=axes_flat[9],
               title=r"Normalized Residual Map $1\sigma$",
               colormap=colormap, vmin=-1.0, vmax=1.0)

    plot_array(array=fit.chi_squared_map, ax=axes_flat[10], title="Chi-Squared Map",
               colormap=colormap, use_log10=True)

    _plot_source_plane(fit, axes_flat[11], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap, use_log10=True)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename=f"subplot_fit_log10{plane_index_tag}", format=output_format)


def subplot_fit_log10_x1_plane(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """6-panel log10 subplot for a single-plane tracer fit."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes_flat = list(axes.flatten())

    try:
        vmax = float(np.max(fit.model_images_of_planes_list[0].array))
    except (IndexError, AttributeError, ValueError):
        vmax = None

    plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap,
               vmax=vmax, use_log10=True)

    try:
        plot_array(array=fit.signal_to_noise_map, ax=axes_flat[1],
                   title="Signal-To-Noise Map", colormap=colormap, use_log10=True)
    except ValueError:
        axes_flat[1].axis("off")

    plot_array(array=fit.model_data, ax=axes_flat[2], title="Model Image",
               colormap=colormap, vmax=vmax, use_log10=True)

    norm_resid = fit.normalized_residual_map
    plot_array(array=norm_resid, ax=axes_flat[3], title="Lens Light Subtracted",
               colormap=colormap)
    _abs_max = _symmetric_vmax(norm_resid)
    plot_array(array=norm_resid, ax=axes_flat[4], title="Normalized Residual Map",
               colormap=colormap, vmin=-_abs_max, vmax=_abs_max)
    plot_array(array=fit.chi_squared_map, ax=axes_flat[5], title="Chi-Squared Map",
               colormap=colormap, use_log10=True)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_fit_log10", format=output_format)


def subplot_of_planes(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    plane_index: Optional[int] = None,
):
    """4-panel subplot per plane: data, subtracted, model image, plane image."""
    if plane_index is None:
        plane_indexes = range(len(fit.tracer.planes))
    else:
        plane_indexes = [plane_index]

    for pidx in plane_indexes:
        fig, axes = plt.subplots(1, 4, figsize=(28, 7))
        axes_flat = list(axes.flatten())

        plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap)

        try:
            subtracted = fit.subtracted_images_of_planes_list[pidx]
            plot_array(array=subtracted, ax=axes_flat[1],
                       title=f"Subtracted Image Plane {pidx}", colormap=colormap)
        except (IndexError, AttributeError):
            axes_flat[1].axis("off")

        try:
            model_img = fit.model_images_of_planes_list[pidx]
            plot_array(array=model_img, ax=axes_flat[2],
                       title=f"Model Image Plane {pidx}", colormap=colormap)
        except (IndexError, AttributeError):
            axes_flat[2].axis("off")

        _plot_source_plane(fit, axes_flat[3], pidx, colormap=colormap)

        plt.tight_layout()
        save_figure(fig, path=output_path, filename=f"subplot_of_plane_{pidx}", format=output_format)


def subplot_tracer_from_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """9-panel tracer subplot derived from a FitImaging object."""
    final_plane_index = len(fit.tracer.planes) - 1

    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    axes_flat = list(axes.flatten())

    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    plot_array(array=fit.model_data, ax=axes_flat[0], title="Model Image",
               colormap=colormap)

    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
        source_vmax = float(np.max(source_model_img.array))
        plot_array(array=source_model_img, ax=axes_flat[1], title="Source Model Image",
                   colormap=colormap, vmax=source_vmax)
    except (IndexError, AttributeError, ValueError):
        axes_flat[1].axis("off")

    _plot_source_plane(fit, axes_flat[2], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap)

    # Lens plane mass quantities (log10)
    zoom = aa.Zoom2D(mask=fit.mask)
    grid = aa.Grid2D.from_extent(
        extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
    )

    tan_cc, rad_cc = _critical_curves_from(tracer, grid)
    image_plane_lines = _to_lines(tan_cc, rad_cc)

    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])
    plot_array(array=lens_image, ax=axes_flat[3], title="Lens Image",
               lines=image_plane_lines, colormap=colormap, use_log10=True)

    for i in range(4, 9):
        axes_flat[i].axis("off")

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_tracer", format=output_format)


def subplot_fit_combined(
    fit_list: List,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """Combined multi-row subplot for a list of FitImaging objects."""
    n_fits = len(fit_list)
    n_cols = 6
    fig, axes = plt.subplots(n_fits, n_cols, figsize=(7 * n_cols, 7 * n_fits))
    if n_fits == 1:
        all_axes = [list(axes)]
    else:
        all_axes = [list(axes[i]) for i in range(n_fits)]

    final_plane_index = len(fit_list[0].tracer.planes) - 1

    for row, fit in enumerate(fit_list):
        row_axes = all_axes[row]

        plot_array(array=fit.data, ax=row_axes[0], title="Data", colormap=colormap)

        try:
            subtracted = fit.subtracted_images_of_planes_list[1]
            plot_array(array=subtracted, ax=row_axes[1], title="Subtracted Image",
                       colormap=colormap)
        except (IndexError, AttributeError):
            row_axes[1].axis("off")

        try:
            lens_model = fit.model_images_of_planes_list[0]
            plot_array(array=lens_model, ax=row_axes[2], title="Lens Model Image",
                       colormap=colormap)
        except (IndexError, AttributeError):
            row_axes[2].axis("off")

        try:
            source_model = fit.model_images_of_planes_list[final_plane_index]
            plot_array(array=source_model, ax=row_axes[3], title="Source Model Image",
                       colormap=colormap)
        except (IndexError, AttributeError):
            row_axes[3].axis("off")

        try:
            _plot_source_plane(fit, row_axes[4], final_plane_index, colormap=colormap)
        except Exception:
            row_axes[4].axis("off")

        plot_array(array=fit.normalized_residual_map, ax=row_axes[5],
                   title="Normalized Residual Map", colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_fit_combined", format=output_format)


def subplot_fit_combined_log10(
    fit_list: List,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """Combined log10 multi-row subplot for a list of FitImaging objects."""
    n_fits = len(fit_list)
    n_cols = 6
    fig, axes = plt.subplots(n_fits, n_cols, figsize=(7 * n_cols, 7 * n_fits))
    if n_fits == 1:
        all_axes = [list(axes)]
    else:
        all_axes = [list(axes[i]) for i in range(n_fits)]

    final_plane_index = len(fit_list[0].tracer.planes) - 1

    for row, fit in enumerate(fit_list):
        row_axes = all_axes[row]

        plot_array(array=fit.data, ax=row_axes[0], title="Data", colormap=colormap,
                   use_log10=True)

        try:
            subtracted = fit.subtracted_images_of_planes_list[1]
            plot_array(array=subtracted, ax=row_axes[1], title="Subtracted Image",
                       colormap=colormap, use_log10=True)
        except (IndexError, AttributeError):
            row_axes[1].axis("off")

        try:
            lens_model = fit.model_images_of_planes_list[0]
            plot_array(array=lens_model, ax=row_axes[2], title="Lens Model Image",
                       colormap=colormap, use_log10=True)
        except (IndexError, AttributeError):
            row_axes[2].axis("off")

        try:
            source_model = fit.model_images_of_planes_list[final_plane_index]
            plot_array(array=source_model, ax=row_axes[3], title="Source Model Image",
                       colormap=colormap, use_log10=True)
        except (IndexError, AttributeError):
            row_axes[3].axis("off")

        try:
            _plot_source_plane(fit, row_axes[4], final_plane_index, colormap=colormap,
                               use_log10=True)
        except Exception:
            row_axes[4].axis("off")

        plot_array(array=fit.normalized_residual_map, ax=row_axes[5],
                   title="Normalized Residual Map", colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="fit_combined_log10", format=output_format)


def _symmetric_vmax(array) -> float:
    """Return abs-max finite value for symmetric colormap scaling."""
    try:
        vals = _zoom_array_2d(array).native.array
    except AttributeError:
        vals = np.asarray(array)
    finite = vals[np.isfinite(vals)]
    return float(np.max(np.abs(finite))) if finite.size else 1.0
