import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag

from autolens.plot.plot_utils import (
    plot_array,
    _to_lines,
    _save_subplot,
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
    if source_vmax is not None:
        from autoarray.structures.plot.structure_plotters import _zoom_array
        import matplotlib as mpl
        _ax = axes_flat[1]
        _plot_with_vmax(fit.data, _ax, "Data (Source Scale)", colormap, vmax=source_vmax)
    else:
        plot_array(array=fit.data, ax=axes_flat[1], title="Data (Source Scale)",
                   colormap=colormap)

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
        if source_vmax is not None:
            _plot_with_vmin_vmax(subtracted_img, axes_flat[5],
                                 "Lens Light Subtracted", colormap, vmin=0.0, vmax=source_vmax)
        else:
            plot_array(array=subtracted_img, ax=axes_flat[5],
                       title="Lens Light Subtracted", colormap=colormap)
    else:
        axes_flat[5].axis("off")

    # Source model image at source scale
    try:
        source_model_img = fit.model_images_of_planes_list[final_plane_index]
    except (IndexError, AttributeError):
        source_model_img = None
    if source_model_img is not None:
        if source_vmax is not None:
            _plot_with_vmax(source_model_img, axes_flat[6], "Source Model Image",
                            colormap, vmax=source_vmax)
        else:
            plot_array(array=source_model_img, ax=axes_flat[6],
                       title="Source Model Image", colormap=colormap)
    else:
        axes_flat[6].axis("off")

    # Source plane zoomed
    _plot_source_plane(fit, axes_flat[7], final_plane_index, zoom_to_brightest=True,
                       colormap=colormap)

    # Normalized residual map (symmetric)
    norm_resid = fit.normalized_residual_map
    _plot_symmetric(norm_resid, axes_flat[8], "Normalized Residual Map", colormap)

    # Normalized residual map clipped to [-1, 1]
    _plot_with_vmin_vmax(norm_resid, axes_flat[9],
                         r"Normalized Residual Map $1\sigma$", colormap,
                         vmin=-1.0, vmax=1.0)

    plot_array(array=fit.chi_squared_map, ax=axes_flat[10],
               title="Chi-Squared Map", colormap=colormap)

    # Source plane not zoomed
    _plot_source_plane(fit, axes_flat[11], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap)

    plt.tight_layout()
    _save_subplot(fig, output_path, f"subplot_fit{plane_index_tag}", output_format)


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

    if vmax is not None:
        _plot_with_vmax(fit.data, axes_flat[0], "Data", colormap, vmax=vmax)
    else:
        plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap)

    plot_array(array=fit.signal_to_noise_map, ax=axes_flat[1],
               title="Signal-To-Noise Map", colormap=colormap)

    if vmax is not None:
        _plot_with_vmax(fit.model_data, axes_flat[2], "Model Image", colormap, vmax=vmax)
    else:
        plot_array(array=fit.model_data, ax=axes_flat[2], title="Model Image",
                   colormap=colormap)

    norm_resid = fit.normalized_residual_map
    plot_array(array=norm_resid, ax=axes_flat[3], title="Lens Light Subtracted",
               colormap=colormap)

    _plot_with_vmin(norm_resid, axes_flat[4], "Subtracted Image Zero Minimum",
                    colormap, vmin=0.0)

    _plot_symmetric(norm_resid, axes_flat[5], "Normalized Residual Map", colormap)

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit_x1_plane", output_format)


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
    _plot_symmetric(norm_resid, axes_flat[8], "Normalized Residual Map", colormap)

    _plot_with_vmin_vmax(norm_resid, axes_flat[9],
                         r"Normalized Residual Map $1\sigma$", colormap,
                         vmin=-1.0, vmax=1.0)

    plot_array(array=fit.chi_squared_map, ax=axes_flat[10], title="Chi-Squared Map",
               colormap=colormap, use_log10=True)

    _plot_source_plane(fit, axes_flat[11], final_plane_index, zoom_to_brightest=False,
                       colormap=colormap, use_log10=True)

    plt.tight_layout()
    _save_subplot(fig, output_path, f"subplot_fit_log10{plane_index_tag}", output_format)


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

    if vmax is not None:
        _plot_with_vmax(fit.data, axes_flat[0], "Data", colormap, vmax=vmax,
                        use_log10=True)
    else:
        plot_array(array=fit.data, ax=axes_flat[0], title="Data", colormap=colormap,
                   use_log10=True)

    try:
        plot_array(array=fit.signal_to_noise_map, ax=axes_flat[1],
                   title="Signal-To-Noise Map", colormap=colormap, use_log10=True)
    except ValueError:
        axes_flat[1].axis("off")

    if vmax is not None:
        _plot_with_vmax(fit.model_data, axes_flat[2], "Model Image", colormap,
                        vmax=vmax, use_log10=True)
    else:
        plot_array(array=fit.model_data, ax=axes_flat[2], title="Model Image",
                   colormap=colormap, use_log10=True)

    norm_resid = fit.normalized_residual_map
    plot_array(array=norm_resid, ax=axes_flat[3], title="Lens Light Subtracted",
               colormap=colormap)
    _plot_symmetric(norm_resid, axes_flat[4], "Normalized Residual Map", colormap)
    plot_array(array=fit.chi_squared_map, ax=axes_flat[5], title="Chi-Squared Map",
               colormap=colormap, use_log10=True)

    plt.tight_layout()
    _save_subplot(fig, output_path, "subplot_fit_log10", output_format)


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
        _save_subplot(fig, output_path, f"subplot_of_plane_{pidx}", output_format)


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
        _plot_with_vmax(source_model_img, axes_flat[1], "Source Model Image",
                        colormap, vmax=source_vmax)
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
    _save_subplot(fig, output_path, "subplot_tracer", output_format)


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
    _save_subplot(fig, output_path, "subplot_fit_combined", output_format)


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
    _save_subplot(fig, output_path, "fit_combined_log10", output_format)


# ---------------------------------------------------------------------------
# Private helpers for vmin/vmax manipulation without Cmap objects
# ---------------------------------------------------------------------------

def _plot_with_vmax(array, ax, title, colormap, vmax, use_log10=False):
    from autoarray.plot.plots.array import plot_array as _aa_plot_array
    from autoarray.structures.plot.structure_plotters import (
        _auto_mask_edge, _zoom_array,
    )
    array = _zoom_array(array)
    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None
    mask = _auto_mask_edge(array) if hasattr(array, "mask") else None
    _aa_plot_array(array=arr, ax=ax, extent=extent, mask=mask,
                   title=title, colormap=colormap, use_log10=use_log10,
                   vmax=vmax, structure=array)


def _plot_with_vmin(array, ax, title, colormap, vmin, use_log10=False):
    from autoarray.plot.plots.array import plot_array as _aa_plot_array
    from autoarray.structures.plot.structure_plotters import (
        _auto_mask_edge, _zoom_array,
    )
    array = _zoom_array(array)
    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None
    mask = _auto_mask_edge(array) if hasattr(array, "mask") else None
    _aa_plot_array(array=arr, ax=ax, extent=extent, mask=mask,
                   title=title, colormap=colormap, use_log10=use_log10,
                   vmin=vmin, structure=array)


def _plot_with_vmin_vmax(array, ax, title, colormap, vmin, vmax, use_log10=False):
    from autoarray.plot.plots.array import plot_array as _aa_plot_array
    from autoarray.structures.plot.structure_plotters import (
        _auto_mask_edge, _zoom_array,
    )
    array = _zoom_array(array)
    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None
    mask = _auto_mask_edge(array) if hasattr(array, "mask") else None
    _aa_plot_array(array=arr, ax=ax, extent=extent, mask=mask,
                   title=title, colormap=colormap, use_log10=use_log10,
                   vmin=vmin, vmax=vmax, structure=array)


def _plot_symmetric(array, ax, title, colormap):
    """Plot with symmetric colormap (vmin = -vmax)."""
    from autoarray.structures.plot.structure_plotters import _zoom_array
    _arr = _zoom_array(array)
    try:
        vals = _arr.native.array
    except AttributeError:
        vals = np.asarray(_arr)
    abs_max = float(np.max(np.abs(vals[np.isfinite(vals)]))) if np.any(np.isfinite(vals)) else 1.0
    _plot_with_vmin_vmax(array, ax, title, colormap, vmin=-abs_max, vmax=abs_max)
