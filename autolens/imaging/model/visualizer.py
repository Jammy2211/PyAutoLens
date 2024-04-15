from autoarray import exc

import autofit as af
import autogalaxy as ag

from autolens.imaging.model.plotter_interface import PlotterInterfaceImaging
from autolens import exc

class VisualizerImaging(af.Visualizer):

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
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        plotter_interface = PlotterInterfaceImaging(image_path=paths.image_path)

        plotter_interface.imaging(dataset=analysis.dataset)

        if analysis.positions_likelihood is not None:
            plotter_interface.image_with_positions(
                image=analysis.dataset.data,
                positions=analysis.positions_likelihood.positions,
            )

        if analysis.adapt_images is not None:
            plotter_interface.adapt_images(
                adapt_images=analysis.adapt_images
            )

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Tracer`, including the images of each of its galaxies.

        - Images of the best-fit `FitImaging`, including the model-image, residuals and chi-squared of its fit to
          the imaging data.

        - The adapt-images of the model-fit showing how the galaxies are used to represent different galaxies in
          the dataset.

        The images output by this function are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        fit = analysis.fit_from(instance=instance)

        if analysis.positions_likelihood is not None:
            analysis.positions_likelihood.output_positions_info(
                output_path=paths.output_path, tracer=fit.tracer
            )

        if fit.inversion is not None:
            try:
                fit.inversion.reconstruction
            except exc.InversionException:
                return

        plotter_interface = PlotterInterfaceImaging(image_path=paths.image_path)

        try:
            plotter_interface.fit_imaging(fit=fit, during_analysis=during_analysis)
        except exc.InversionException:
            pass

        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        extent = fit.data.extent_of_zoomed_array(buffer=0)
        shape_native = fit.data.zoomed_around_mask(buffer=0).shape_native

        grid = ag.Grid2D.from_extent(
            extent=extent,
            shape_native=shape_native
        )

        plotter_interface.tracer(
            tracer=tracer, grid=grid, during_analysis=during_analysis
        )
        plotter_interface.galaxies(
            galaxies=tracer.galaxies, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            if fit.inversion.has(cls=ag.AbstractMapper):
                plotter_interface.inversion(
                    inversion=fit.inversion, during_analysis=during_analysis
                )


     