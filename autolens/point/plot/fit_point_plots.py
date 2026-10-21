import numpy as np
from typing import Optional

from autoarray.plot.utils import subplots, save_figure, conf_subplot_figsize, tight_layout


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

    The subplot contains one or two panels depending on whether flux
    measurements are present in the dataset:

    * **Positions panel** (always shown), titled ``Fit Positions (<name>)``.
      Drawn on the same frame as the dataset positions panel (the
      ``PointSolver`` grid extent, no axis labels), it shows:

      - the observed positions as large filled markers with their noise-map
        circles and the sigma in the legend;
      - the model positions on top as red crosses;
      - a thin ``Residual`` line from each observed position to the model
        position the fit's own pairing matches it to (Hungarian for
        ``FitPositionsImagePair``, nearest for ``FitPositionsImagePairRepeat``
        and the dominant nearest term of ``FitPositionsImagePairAll``);
      - model images no observed position is paired with as hollow grey
        crosses (``Unmatched model image``);
      - the tangential / radial critical curves, thin and light, clipped at the
        panel edge where they extend beyond the solver grid;
      - the centre of every mass profile in the tracer (``Mass centre``).

      For a source-plane fit (e.g. ``FitPositionsSource``) the model data are
      the observed positions ray-traced to the source plane: they are drawn as
      ``Model (source plane)`` crosses and no residual lines or unmatched images
      are drawn, as there is no image-plane pairing.
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
        Accepted for a uniform plotter interface; not drawn.
    title_prefix : str, optional
        Text prepended to every panel title.
    solver : PointSolver, optional
        The solver whose grid sets the positions panel extent. Defaults to
        ``fit.solver``.
    """
    from autoarray.plot.yx import plot_yx
    from autolens.point.plot.positions_panel import (
        finish_positions_panel,
        plot_critical_curves,
        plot_fit_model_positions,
        plot_mass_centres,
        plot_observed_positions,
        positions_extent_from,
    )

    has_fluxes = fit.dataset.fluxes is not None
    n = 2 if has_fluxes else 1

    if solver is None:
        solver = getattr(fit, "solver", None)

    if image_plane_lines is None and source_plane_lines is None:
        from autolens.lens.plot.critical_curves import (
            _compute_critical_curve_lines,
            critical_curve_grid_from,
        )

        image_plane_lines, image_plane_line_colors, _, _ = (
            _compute_critical_curve_lines(
                fit.tracer, critical_curve_grid_from(dataset=fit.dataset, solver=solver)
            )
        )

    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    _prefix = f"{title_prefix.rstrip()} " if title_prefix else ""

    # Positions panel
    ax = axes_flat[0]
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
        extent=positions_extent_from(positions=fit.dataset.positions, solver=solver),
    )

    # Fluxes panel
    if has_fluxes and n > 1:
        y = np.array(fit.dataset.fluxes)
        x = np.arange(len(y))
        plot_yx(
            y=y,
            x=x,
            ax=axes_flat[1],
            title=f"{_prefix}Fit Fluxes ({fit.dataset.name})",
            output_path=None,
            output_filename="fit_point_fluxes",
            output_format=output_format,
        )

    tight_layout()
    save_figure(fig, path=output_path, filename="fit", format=output_format)


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
