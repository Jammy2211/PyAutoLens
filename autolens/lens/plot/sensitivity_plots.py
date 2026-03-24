"""Standalone subplot functions for subhalo sensitivity mapping visualisation."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import autoarray as aa

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.utils import save_figure


def subplot_tracer_images(
    mask,
    tracer_perturb,
    tracer_no_perturb,
    source_image,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
):
    """6-panel subplot showing lensed images and residuals from a perturbed tracer."""
    from autolens.lens.tracer_util import critical_curves_from, caustics_from
    from autolens.plot.plot_utils import _to_lines

    grid = aa.Grid2D.from_mask(mask=mask)

    image = tracer_perturb.image_2d_from(grid=grid)
    lensed_source_image = tracer_perturb.image_2d_via_input_plane_image_from(
        grid=grid, plane_image=source_image
    )
    lensed_source_image_no_perturb = tracer_no_perturb.image_2d_via_input_plane_image_from(
        grid=grid, plane_image=source_image
    )

    unmasked_grid = mask.derive_grid.unmasked

    try:
        tan_cc_p, rad_cc_p = critical_curves_from(tracer=tracer_perturb, grid=unmasked_grid)
        perturb_cc_lines = _to_lines(list(tan_cc_p), list(rad_cc_p))
    except Exception:
        perturb_cc_lines = None

    try:
        tan_ca_p, rad_ca_p = caustics_from(tracer=tracer_perturb, grid=unmasked_grid)
        perturb_ca_lines = _to_lines(list(tan_ca_p), list(rad_ca_p))
    except Exception:
        perturb_ca_lines = None

    try:
        tan_cc_n, rad_cc_n = critical_curves_from(tracer=tracer_no_perturb, grid=unmasked_grid)
        no_perturb_cc_lines = _to_lines(list(tan_cc_n), list(rad_cc_n))
    except Exception:
        no_perturb_cc_lines = None

    residual_map = lensed_source_image - lensed_source_image_no_perturb

    fig, axes = plt.subplots(1, 6, figsize=(42, 7))

    plot_array(array=image, ax=axes[0], title="Image",
               colormap=colormap, use_log10=use_log10)
    plot_array(array=lensed_source_image, ax=axes[1], title="Lensed Source Image",
               colormap=colormap, use_log10=use_log10, lines=perturb_cc_lines)
    plot_array(array=source_image, ax=axes[2], title="Source Image",
               colormap=colormap, use_log10=use_log10, lines=perturb_ca_lines)
    plot_array(array=tracer_perturb.convergence_2d_from(grid=grid), ax=axes[3],
               title="Convergence", colormap=colormap, use_log10=use_log10)
    plot_array(array=lensed_source_image, ax=axes[4],
               title="Lensed Source Image (No Subhalo)",
               colormap=colormap, use_log10=use_log10, lines=no_perturb_cc_lines)
    plot_array(array=residual_map, ax=axes[5],
               title="Residual Map (Subhalo - No Subhalo)",
               colormap=colormap, use_log10=use_log10, lines=no_perturb_cc_lines)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_lensed_images", format=output_format)


def subplot_sensitivity(
    result,
    data_subtracted,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
):
    """8-panel sensitivity subplot: log-likelihood/evidence maps and above-threshold map."""
    log_likelihoods = result.figure_of_merit_array(
        use_log_evidences=False,
        remove_zeros=True,
    )

    try:
        log_evidences = result.figure_of_merit_array(
            use_log_evidences=True,
            remove_zeros=True,
        )
    except TypeError:
        log_evidences = np.zeros_like(log_likelihoods)

    above_threshold = np.where(log_likelihoods > 5.0, 1.0, 0.0)
    above_threshold = aa.Array2D(values=above_threshold, mask=log_likelihoods.mask)

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    axes_flat = list(axes.flatten())

    plot_array(array=data_subtracted, ax=axes_flat[0], title="Subtracted Image",
               colormap=colormap, use_log10=use_log10)
    plot_array(array=log_evidences, ax=axes_flat[1], title="Increase in Log Evidence",
               colormap=colormap)
    plot_array(array=log_likelihoods, ax=axes_flat[2], title="Increase in Log Likelihood",
               colormap=colormap)
    plot_array(array=above_threshold, ax=axes_flat[3], title="Log Likelihood > 5.0",
               colormap=colormap)

    ax_idx = 4
    try:
        log_evidences_base = result._array_2d_from(result.log_evidences_base)
        log_evidences_perturbed = result._array_2d_from(result.log_evidences_perturbed)

        base_vals = np.asarray(log_evidences_base)
        perturb_vals = np.asarray(log_evidences_perturbed)
        finite_base = base_vals[np.isfinite(base_vals) & (base_vals != 0)]
        finite_perturb = perturb_vals[np.isfinite(perturb_vals) & (perturb_vals != 0)]
        if len(finite_base) > 0 and len(finite_perturb) > 0:
            vmin = float(np.min([np.min(finite_base), np.min(finite_perturb)]))
            vmax = float(np.max([np.max(finite_base), np.max(finite_perturb)]))
        else:
            vmin = vmax = None

        plot_array(array=log_evidences_base, ax=axes_flat[ax_idx],
                   title="Log Evidence Base", colormap=colormap, vmin=vmin, vmax=vmax)
        ax_idx += 1
        plot_array(array=log_evidences_perturbed, ax=axes_flat[ax_idx],
                   title="Log Evidence Perturb", colormap=colormap, vmin=vmin, vmax=vmax)
        ax_idx += 1
    except (TypeError, AttributeError):
        pass

    try:
        log_likelihoods_base = result._array_2d_from(result.log_likelihoods_base)
        log_likelihoods_perturbed = result._array_2d_from(result.log_likelihoods_perturbed)

        base_vals = np.asarray(log_likelihoods_base)
        perturb_vals = np.asarray(log_likelihoods_perturbed)
        finite_base = base_vals[np.isfinite(base_vals) & (base_vals != 0)]
        finite_perturb = perturb_vals[np.isfinite(perturb_vals) & (perturb_vals != 0)]
        if len(finite_base) > 0 and len(finite_perturb) > 0:
            vmin = float(np.min([np.min(finite_base), np.min(finite_perturb)]))
            vmax = float(np.max([np.max(finite_base), np.max(finite_perturb)]))
        else:
            vmin = vmax = None

        if ax_idx < len(axes_flat):
            plot_array(array=log_likelihoods_base, ax=axes_flat[ax_idx],
                       title="Log Likelihood Base", colormap=colormap, vmin=vmin, vmax=vmax)
            ax_idx += 1
        if ax_idx < len(axes_flat):
            plot_array(array=log_likelihoods_perturbed, ax=axes_flat[ax_idx],
                       title="Log Likelihood Perturb", colormap=colormap, vmin=vmin, vmax=vmax)
    except (TypeError, AttributeError):
        pass

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_sensitivity", format=output_format)


def subplot_figures_of_merit_grid(
    result,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log_evidences: bool = True,
    remove_zeros: bool = True,
):
    """Single-panel subplot: the figures-of-merit grid for sensitivity mapping."""
    figures_of_merit = result.figure_of_merit_array(
        use_log_evidences=use_log_evidences,
        remove_zeros=remove_zeros,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_array(array=figures_of_merit, ax=ax, title="Increase in Log Evidence",
               colormap=colormap)
    plt.tight_layout()
    save_figure(fig, path=output_path, filename="sensitivity", format=output_format)
