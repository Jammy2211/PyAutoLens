import autofit as af

from autolens.point.model.plotter import PlotterPoint
from autolens.lens.plot.critical_curves import critical_curve_grid_from
from autolens.imaging.plot.fit_imaging_plots import _compute_critical_curve_lines


class VisualizerPoint(af.Visualizer):
    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        plotter = PlotterPoint(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter.dataset_point(
            dataset=analysis.dataset, solver=getattr(analysis, "solver", None)
        )

        if getattr(analysis, "image", None) is not None:
            # Dataset-only figure (observed positions, no model): `image_with_points.png`.
            plotter.image_with_points(image=analysis.image, dataset=analysis.dataset)

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
        quick_update: bool = False,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Tracer`, including the images of each of its galaxies.

        - Images of the best-fit `FitPointDataset`, including the model-image, residuals and chi-squared of its fit to
          the imaging data.

        The images output by this function are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        """
        fit = analysis.fit_for_visualization(instance=instance)

        plotter = PlotterPoint(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        tracer = fit.tracer

        grid = critical_curve_grid_from(dataset=fit.dataset)

        # Compute critical curves and caustics once for all plot functions, on a grid which also covers the
        # solver grid, so the curves on the solver-extent positions panel are only ever cut by its frame.
        ip_lines, ip_colors, sp_lines, sp_colors = _compute_critical_curve_lines(
            tracer,
            critical_curve_grid_from(
                dataset=fit.dataset, solver=getattr(fit, "solver", None)
            ),
        )

        plotter.fit_point(
            fit=fit,
            quick_update=quick_update,
            image_plane_lines=ip_lines,
            image_plane_line_colors=ip_colors,
            source_plane_lines=sp_lines,
            source_plane_line_colors=sp_colors,
        )

        # Output even on `quick_update`, so the observed image with model positions updates on-the-fly. Saved as
        # `fit_image_with_points.png`, so the dataset-only `image_with_points.png` from before the fit is kept.
        if getattr(analysis, "image", None) is not None:
            plotter.image_with_points(
                image=analysis.image,
                dataset=fit.dataset,
                fit=fit,
                image_plane_lines=ip_lines,
                image_plane_line_colors=ip_colors,
            )

        if quick_update:
            return

        plotter.tracer(
            tracer=tracer,
            grid=grid,
            image_plane_lines=ip_lines,
            image_plane_line_colors=ip_colors,
            source_plane_lines=sp_lines,
            source_plane_line_colors=sp_colors,
        )
        plotter.galaxies(
            galaxies=tracer.galaxies,
            grid=grid,
        )

        # Off by default (`fit_point_dataset: subplot_point_solver`): re-solves the best-fit model with NumPy to show
        # the solver's refinement steps, so it is output only for the full (not quick-update) visualization.
        plotter.point_solver(
            fit=fit, image_plane_lines=ip_lines if ip_lines is not None else []
        )
