import logging

import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout

logger = logging.getLogger(__name__)


def subplot_fit(
    fit,
    output_path: Optional[str] = None,
    output_format: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    source_plane_lines=None,
    source_plane_line_colors=None,
    title_prefix: str = None,
    solver=None,
):
    """
    Produce a subplot summarising a `FitPointDataset`.

    The subplot is one row of two or three panels, depending on whether flux measurements are present in the
    dataset:

    * **Positions panel** (always shown), titled ``Fit Positions (<name>)``.
      Drawn on the same frame as the dataset positions panel (the
      ``PointSolver`` grid extent, no axis labels), it shows:

      - the observed positions as large filled markers with their noise-map
        circles and the sigma in the legend, numbered 1..N in the order the
        ``PointSolver`` figure numbers its images, each number beside its image;
      - for an image-plane fit, the model positions on top as red crosses, a
        thin ``Residual`` line from each observed position to the model
        position the fit's own pairing matches it to (Hungarian for
        ``FitPositionsImagePair``, nearest for ``FitPositionsImagePairRepeat``
        and the dominant nearest term of ``FitPositionsImagePairAll``), and
        model images no observed position is paired with as hollow grey
        crosses (``Unmatched model image``);
      - the tangential / radial critical curves, thin and light, clipped at the
        panel edge where they extend beyond the solver grid;
      - the centre of every mass profile in the tracer (``Mass centre``).

      A source-plane fit (e.g. ``FitPositionsSource``) draws no model positions here: its model data lie in the
      source plane and are drawn on the source-plane panel.
    * **Source-plane panel** (always shown), titled ``Fit Source Plane (<name>)``: the observed positions
      ray-traced to the point source's plane, numbered as on the positions panel, the source centre and the
      caustics of that plane, zoomed onto the traced positions with an inset of the whole caustic. What it draws is
      paired to the fit class (see :mod:`autolens.point.plot.source_plane_panel`): context only for an image-plane
      chi-squared; the noise footprint and chi-squared of the fit's weighting for a source-plane chi-squared.
    * **Fluxes panel** (shown only when ``fit.dataset.fluxes`` is not
      ``None``): a bar/line plot of the observed flux values.

    Parameters
    ----------
    fit : FitPointDataset
        The point-source dataset fit to visualise.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    image_plane_lines : list, optional
        Critical-curve lines drawn on the positions panel. If this and
        ``source_plane_lines`` are both ``None`` they are computed from the
        fit's tracer on ``critical_curve_grid_from`` (enlarged to cover the
        solver grid); pass ``[]`` to draw none.
    image_plane_line_colors : list, optional
        Colours of each image-plane line (``"white"`` tangential, ``"yellow"``
        radial, as returned by the critical-curve helper).
    source_plane_lines, source_plane_line_colors
        Caustics drawn on the source-plane panel when the point source is in the tracer's last plane (the plane
        these lines are computed for). For a point source in an earlier plane of a multi-plane tracer, or when no
        lines are passed, the caustics of the point source's own plane are computed on
        ``critical_curve_grid_from``. Pass ``[]`` for both line arguments to draw no curves on either panel.
    title_prefix : str, optional
        Text prepended to every panel title.
    solver : PointSolver, optional
        The solver whose grid sets the positions panel extent. Defaults to
        ``fit.solver``.
    """
    from autoarray.plot.yx import plot_yx
    from autolens.lens.plot.critical_curves import critical_curve_grid_from
    from autolens.point.plot.positions_panel import (
        finish_positions_panel,
        image_numbers_from,
        plot_critical_curves,
        plot_fit_model_positions,
        plot_image_numbers,
        plot_mass_centres,
        plot_observed_positions,
        positions_extent_from,
    )
    from autolens.point.plot import source_plane_panel

    has_fluxes = fit.dataset.fluxes is not None
    n = 3 if has_fluxes else 2

    if solver is None:
        solver = getattr(fit, "solver", None)

    lines_given = image_plane_lines is not None or source_plane_lines is not None

    if not lines_given:
        from autolens.lens.plot.critical_curves import _compute_critical_curve_lines

        (
            image_plane_lines,
            image_plane_line_colors,
            source_plane_lines,
            source_plane_line_colors,
        ) = _compute_critical_curve_lines(
            fit.tracer, critical_curve_grid_from(dataset=fit.dataset, solver=solver)
        )

    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = list(np.array(axes).flatten())

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""

    numbers = image_numbers_from(fit.dataset.positions)

    # Positions panel
    ax = axes_flat[0]
    extent = positions_extent_from(positions=fit.dataset.positions, solver=solver)
    plot_critical_curves(ax, image_plane_lines, image_plane_line_colors)
    plot_mass_centres(ax, fit.tracer)
    plot_observed_positions(
        ax=ax,
        positions=fit.dataset.positions,
        noise_map=fit.dataset.positions_noise_map,
    )
    plot_fit_model_positions(ax, fit.positions)
    finish_positions_panel(
        ax=ax,
        title=f"{_prefix}Fit Positions ({fit.dataset.name})",
        extent=extent,
    )
    # The source-plane panel's title is lifted above its subtitle: lift this one equally so the titles align.
    ax.set_title(
        ax.get_title(),
        fontsize=ax.title.get_fontsize(),
        pad=source_plane_panel.TITLE_PAD,
    )
    plot_image_numbers(ax, fit.dataset.positions, numbers, extent)

    # Source-plane panel
    source_plane_title = f"{_prefix}Fit Source Plane ({fit.dataset.name})"
    try:
        data = source_plane_panel.source_plane_data_from(
            fit.positions, numbers=numbers
        )
        tangential, radial = source_plane_caustics_from(
            fit=fit,
            solver=solver,
            lines_given=lines_given,
            source_plane_lines=source_plane_lines,
            source_plane_line_colors=source_plane_line_colors,
        )
        source_plane_panel.plot_source_plane_panel(
            ax=axes_flat[1],
            data=data,
            title=source_plane_title,
            tangential_caustics=tangential,
            radial_caustics=radial,
        )
    except Exception:
        # A visualization must never stop a model-fit: report the failure and leave the panel empty.
        logger.warning(
            "The source-plane panel of fit.png could not be drawn; it is left empty.",
            exc_info=True,
        )
        axes_flat[1].cla()
        axes_flat[1].set_title(source_plane_title)
        axes_flat[1].set_axis_off()

    # Fluxes panel
    if has_fluxes:
        y = np.array(fit.dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[2],
            title=f"{_prefix}Fit Fluxes ({fit.dataset.name})",
            output_path=None,
            output_filename="fit_point_fluxes",
            output_format=output_format,
        )

    tight_layout()
    save_figure(fig, path=output_path, filename="fit", format=output_format)


def source_plane_caustics_from(
    fit,
    solver=None,
    lines_given: bool = False,
    source_plane_lines=None,
    source_plane_line_colors=None,
):
    """
    Returns the ``(tangential, radial)`` caustics drawn on the source-plane panel of ``fit``: those of the plane
    the point source is in.

    The pre-computed ``source_plane_lines`` are the caustics of the tracer's last plane, so they are reused when
    the point source is in that plane. Otherwise (a point in an earlier plane of a multi-plane tracer, or no lines
    passed) the caustics of the point's own plane are computed on ``critical_curve_grid_from``. If lines were given
    and both are empty (``[]``), no caustics are drawn.
    """
    from autolens.lens.plot.critical_curves import critical_curve_grid_from
    from autolens.point.plot import source_plane_panel

    plane_index = fit.positions.plane_index
    last_plane = len(fit.tracer.planes) - 1
    in_last_plane = plane_index in (-1, last_plane)

    if lines_given and source_plane_lines is not None and in_last_plane:
        return source_plane_panel.caustics_from_lines(
            source_plane_lines, source_plane_line_colors
        )
    if lines_given and source_plane_lines is not None and len(source_plane_lines) == 0:
        return [], []

    return source_plane_panel.point_plane_caustics_from(
        tracer=fit.tracer,
        plane_index=plane_index,
        grid=critical_curve_grid_from(dataset=fit.dataset, solver=solver),
    )


def plot_image_with_points(
    image,
    dataset,
    fit=None,
    output_path: Optional[str] = None,
    output_format: str = None,
    image_plane_lines=None,
    image_plane_line_colors=None,
    title_prefix: str = None,
):
    """
    Plot an observed image of a lensed point source (e.g. a quasar) with its point-source positions overlaid.

    The observed positions of the `PointDataset` are drawn as hollow white rings, so they stand out against
    the image without hiding the point-source emission beneath them. When a `FitPointDataset` is passed, the model-predicted positions
    are drawn as red crosses (image-plane fits only, as a source-plane fit's model data lies in the source
    plane), and critical curves can be overlaid via ``image_plane_lines``.

    The function produces two figures, depending on whether a fit is passed:

    - ``fit=None`` (dataset-only): only the observed image and observed positions are drawn, saved as
      ``image_with_points``. Use this to display the data before lens modeling, passing no fit and no
      ``image_plane_lines``.
    - ``fit`` given: the model positions (and critical curves, via ``image_plane_lines``) are added, the title
      becomes "<name> Fit Image With Points" and the figure is saved as ``fit_image_with_points``, so it does not
      overwrite the dataset-only figure.

    The image is used only for visualization: it does not enter the point-source likelihood.

    Parameters
    ----------
    image : Array2D
        The observed image the positions are plotted over.
    dataset : PointDataset
        The point-source dataset whose observed positions are overlaid.
    fit : FitPointDataset, optional
        If given, its model positions are overlaid as red crosses and the figure is saved as
        ``fit_image_with_points``; if ``None`` it is the dataset-only ``image_with_points``.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    image_plane_lines : list, optional
        Critical-curve lines overlaid on the image.
    image_plane_line_colors : list, optional
        Colours for each image-plane line.
    """
    from autogalaxy.util.plot_utils import plot_array
    from autoarray.plot.utils import conf_figsize, _overlay_yx_for_origin

    fig, ax = subplots(1, 1, figsize=conf_figsize("figures"))

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""
    plot_array(
        array=image,
        ax=ax,
        lines=image_plane_lines,
        line_colors=image_plane_line_colors,
        title=f"{_prefix}{dataset.name} {'Image' if fit is None else 'Fit Image'} With Points",
    )

    # Positions are (y,x) arc-seconds: reflect them exactly as `plot_array` does its own overlays, using the
    # extent and origin of the raster it drew, so they stay registered to the image (PyAutoArray#565).
    extent, origin = (ax.images[0].get_extent(), ax.images[0].origin) if ax.images else (None, None)

    def _yx(grid):
        grid = np.array(grid.array if hasattr(grid, "array") else grid).reshape(-1, 2)
        return _overlay_yx_for_origin(grid, extent, origin)

    obs_grid = _yx(dataset.positions)
    ax.scatter(
        obs_grid[:, 1], obs_grid[:, 0],
        s=150, facecolors="none", edgecolors="w", linewidths=2, zorder=6, label="Observed",
    )

    # Only image-plane fits have model positions in the image plane; a source-plane fit's model data is the
    # source centre, which must not be drawn over the image.
    from autolens.point.fit.positions.image.abstract import AbstractFitPositionsImagePair

    if fit is not None and isinstance(fit.positions, AbstractFitPositionsImagePair):
        model_grid = _yx(fit.positions.model_data)
        ax.scatter(
            model_grid[:, 1], model_grid[:, 0],
            s=60, c="r", marker="x", linewidths=2, zorder=7, label="Model",
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    filename = "image_with_points" if fit is None else "fit_image_with_points"
    save_figure(fig, path=output_path, filename=filename, format=output_format)
