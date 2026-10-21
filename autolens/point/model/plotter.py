
from autolens.analysis.plotter import Plotter
from autolens.imaging.plot.fit_imaging_plots import _compute_critical_curve_lines
from autolens.lens.plot.critical_curves import critical_curve_grid_from

from autolens.point.fit.dataset import FitPointDataset
from autolens.point.plot.fit_point_plots import subplot_fit as subplot_fit_point
from autolens.point.plot.fit_point_plots import plot_image_with_points
from autolens.point.dataset import PointDataset
from autolens.point.plot.point_dataset_plots import subplot_dataset
from autolens.point.plot import point_solver_plots

from autolens.analysis.plotter import plot_setting


class PlotterPoint(Plotter):
    def dataset_point(self, dataset: PointDataset, solver=None):
        """
        Output visualization of a `PointDataset` dataset.

        Parameters
        ----------
        dataset
            The point dataset which is visualized.
        solver
            The `PointSolver` of the model-fit, whose grid sets the extent of the positions panel. If `None`, the
            panel extent falls back to the critical-curve region around the positions.
        """

        def should_plot(name):
            return plot_setting(section=["point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_dataset"):
            subplot_dataset(
                dataset,
                output_path=output_path,
                output_format=fmt,
                title_prefix=self.title_prefix,
                solver=solver,
            )

    def image_with_points(
        self,
        image,
        dataset: PointDataset,
        fit: FitPointDataset = None,
        image_plane_lines=None,
        image_plane_line_colors=None,
    ):
        """
        Output an observed image of the lensed point source with the point-source positions overlaid.

        Before the fit (``fit=None``) only the observed positions are drawn, output as `image_with_points.png`
        and controlled by the `point_dataset: image_with_points` setting. After the fit the model positions (and
        critical curves, if given) are also drawn, output as `fit_image_with_points.png` (so both figures coexist)
        and controlled by the `fit_point_dataset: fit_image_with_points` setting.

        Parameters
        ----------
        image
            The observed image of the lensed point source (e.g. a quasar), used only for visualization.
        dataset
            The point dataset whose observed positions are overlaid.
        fit
            The maximum log likelihood `FitPointDataset`, whose model positions are overlaid if given.
        image_plane_lines
            Pre-computed critical-curve lines to overlay on the image.
        image_plane_line_colors
            Colours for each image-plane line.
        """
        if fit is None:
            section, name = ["point_dataset"], "image_with_points"
        else:
            section, name = ["fit", "fit_point_dataset"], "fit_image_with_points"

        if not plot_setting(section=section, name=name):
            return

        plot_image_with_points(
            image=image,
            dataset=dataset,
            fit=fit,
            output_path=str(self.image_path),
            output_format=self.fmt,
            image_plane_lines=image_plane_lines,
            image_plane_line_colors=image_plane_line_colors,
            title_prefix=self.title_prefix,
        )

    def fit_point(
        self,
        fit: FitPointDataset,
        quick_update: bool = False,
        image_plane_lines=None,
        image_plane_line_colors=None,
        source_plane_lines=None,
        source_plane_line_colors=None,
    ):
        """
        Visualizes a `FitPointDataset` object.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search.
        image_plane_lines
            Pre-computed critical-curve lines to overlay on image-plane panels.
        image_plane_line_colors
            Colours for each image-plane line.
        source_plane_lines
            Pre-computed caustic lines to overlay on source-plane panels.
        source_plane_line_colors
            Colours for each source-plane line.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_point_dataset"], name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        # Use pre-computed critical curves if provided, otherwise compute once here.
        if image_plane_lines is None and source_plane_lines is None:
            grid = critical_curve_grid_from(
                dataset=fit.dataset, solver=getattr(fit, "solver", None)
            )
            ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curve_lines(
                fit.tracer, grid
            )
        else:
            ip_lines, ip_colors, sp_lines, sp_colors = (
                image_plane_lines, image_plane_line_colors,
                source_plane_lines, source_plane_line_colors,
            )

        # `subplot_fit` computes the curves itself when it is given none, so pass an explicit empty list when the
        # model has no critical curves, rather than computing them twice.
        if ip_lines is None:
            ip_lines, ip_colors = [], []

        if quick_update:
            subplot_fit_point(
                fit, output_path=output_path, output_format=fmt,
                image_plane_lines=ip_lines,
                image_plane_line_colors=ip_colors,
                source_plane_lines=sp_lines,
                source_plane_line_colors=sp_colors,
                title_prefix=self.title_prefix,
            )
            return

        if should_plot("subplot_fit"):
            subplot_fit_point(
                fit, output_path=output_path, output_format=fmt,
                image_plane_lines=ip_lines,
                image_plane_line_colors=ip_colors,
                source_plane_lines=sp_lines,
                source_plane_line_colors=sp_colors,
                title_prefix=self.title_prefix,
            )

    def point_solver(self, fit: FitPointDataset, image_plane_lines=None):
        """
        Output `point_solver.png`: how the `PointSolver` refines its triangles onto the images of the fit's model.

        The model's point source is re-solved with NumPy (the JAX path pads the steps and, inside `jax.jit`, hides
        them), so this is a diagnostic of the model-fit's own solver at this model: did the grid cover every image
        and how many triangles survive each step. A warning is logged when a step keeps nearly as many triangles as
        the JAX solver's `MAX_CONTAINING_SIZE`, which the JAX path truncates to.

        Controlled by the `fit_point_dataset: subplot_point_solver` setting, which is off by default as the
        re-solve and figure cost a few seconds per visualization (more for a many-galaxy multi-plane cluster). The
        figure is not output for fits without a `PointSolver` (e.g. source-plane fits of a model with no solver).

        Parameters
        ----------
        fit
            The maximum log likelihood `FitPointDataset` of the non-linear search.
        image_plane_lines
            Pre-computed critical-curve lines; `None` computes them.
        """
        try:
            if not plot_setting(
                section=["fit", "fit_point_dataset"], name="subplot_point_solver"
            ):
                return
        except KeyError:
            return

        solver = getattr(fit, "solver", None)
        fit_positions = getattr(fit, "positions", None)
        if solver is None or not hasattr(solver, "steps") or fit_positions is None:
            return

        data = point_solver_plots.point_solver_data_from(
            solver=solver,
            tracer=fit.tracer,
            source_plane_coordinate=fit_positions.source_plane_coordinate,
            positions=fit.dataset.positions,
            plane_redshift=fit_positions.plane_redshift,
            image_plane_lines=image_plane_lines,
        )
        point_solver_plots.warn_if_near_jax_cap(data)
        point_solver_plots.plot_point_solver_from_data(
            data=data,
            output_path=str(self.image_path),
            output_format=self.fmt,
            title_prefix=self.title_prefix,
        )
