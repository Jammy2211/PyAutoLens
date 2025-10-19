import autofit as af
import autogalaxy as ag

from autolens.point.model.plotter_interface import PlotterInterfacePoint


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

        plotter_interface = PlotterInterfacePoint(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter_interface.dataset_point(dataset=analysis.dataset)

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
        fit = analysis.fit_from(instance=instance)

        plotter_interface = PlotterInterfacePoint(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter_interface.fit_point(fit=fit, quick_update=quick_update)

        if quick_update:
            return

        tracer = fit.tracer

        grid = ag.Grid2D.from_extent(
            extent=fit.dataset.extent_from(), shape_native=(100, 100)
        )

        plotter_interface.tracer(
            tracer=tracer,
            grid=grid,
        )
        plotter_interface.galaxies(
            galaxies=tracer.galaxies,
            grid=grid,
        )
