import logging

from autoarray import exc

import autofit as af
import autogalaxy as ag

from autolens.imaging.model.plotter_interface import PlotterInterfaceImaging

from autolens.lens import tracer_util
from autolens import exc

logger = logging.getLogger(__name__)

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
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        plotter_interface = PlotterInterfaceImaging(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter_interface.imaging(dataset=analysis.dataset)

        if analysis.positions_likelihood_list is not None:

            positions_list = []

            for positions_likelihood in analysis.positions_likelihood_list:

                positions_list.append(
                    positions_likelihood.positions
                )

            positions = ag.Grid2DIrregular(positions_list)

            plotter_interface.image_with_positions(
                image=analysis.dataset.data,
                positions=positions,
            )

        if analysis.adapt_images is not None:
            plotter_interface.adapt_images(adapt_images=analysis.adapt_images)

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

        - Images of the best-fit `FitImaging`, including the model-image, residuals and chi-squared of its fit to
          the imaging data.

        - The adapt-images of the model-fit showing how the galaxies are used to represent different galaxies in
          the dataset.

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
        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        visuals_2d_of_planes_list = tracer_util.visuals_2d_of_planes_list_from(
            tracer=fit.tracer,
            grid=fit.grids.lp.mask.derive_grid.all_false
        )

        plotter_interface = PlotterInterfaceImaging(
            image_path=paths.image_path,
            title_prefix=analysis.title_prefix,
        )

        try:
            plotter_interface.fit_imaging(
                fit=fit,
                visuals_2d_of_planes_list=visuals_2d_of_planes_list,
                quick_update=quick_update,
            )
        except exc.InversionException:
            pass

        if quick_update:
            return

        # Full update based on configs.

        if analysis.positions_likelihood_list is not None:

            overwrite_file = True

            for positions_likelihood in analysis.positions_likelihood_list:

                positions_likelihood.output_positions_info(
                    output_path=paths.output_path, tracer=fit.tracer, overwrite_file=overwrite_file
                )

                overwrite_file = False

        if fit.inversion is not None:
            try:
                fit.inversion.reconstruction
            except exc.InversionException:
                logger(
                    ag.exc.invalid_linear_algebra_for_visualization_message()
                )
                return

        zoom = ag.Zoom2D(mask=fit.mask)

        extent = zoom.extent_from(buffer=0)
        shape_native = zoom.shape_native

        grid = ag.Grid2D.from_extent(extent=extent, shape_native=shape_native)

        plotter_interface.tracer(
            tracer=tracer,
            grid=grid,
            visuals_2d_of_planes_list=visuals_2d_of_planes_list
        )
        plotter_interface.galaxies(
            galaxies=tracer.galaxies,
            grid=fit.grids.lp,
        )
        if fit.inversion is not None:
            if fit.inversion.has(cls=ag.AbstractMapper):
                plotter_interface.inversion(
                    inversion=fit.inversion,
                )

    @staticmethod
    def visualize_before_fit_combined(
        analyses,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        Performs visualization before the non-linear search begins of information which shared across all analyses
        on a single matplotlib figure.

        This function outputs visuals of all information which does not vary during the fit, for example the dataset
        being fitted.

        Parameters
        ----------
        analyses
            The list of all analysis objects used for fitting via yhe non-linear search.
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        if analyses is None:
            return

        plotter = PlotterInterfaceImaging(
            image_path=paths.image_path, title_prefix=analyses[0].title_prefix
        )

        dataset_list = [analysis.dataset for analysis in analyses]

        plotter.imaging_combined(
            dataset_list=dataset_list,
        )

    @staticmethod
    def visualize_combined(
        analyses,
        paths: af.AbstractPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
        quick_update: bool = False,
    ):
        """
        Performs visualization during the non-linear search of information which is shared across all analyses on a
        single matplotlib figure.

        This function outputs visuals of all information which varies during the fit, for example the model-fit to
        the dataset being fitted.

        Parameters
        ----------
        analyses
            The list of all analysis objects used for fitting via yhe non-linear search.
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        if analyses is None:
            return

        plotter = PlotterInterfaceImaging(
            image_path=paths.image_path, title_prefix=analyses[0].title_prefix
        )

        fit_list = [
            analysis.fit_from(instance=single_instance)
            for analysis, single_instance in zip(analyses, instance)
        ]

        plotter.fit_imaging_combined(
            fit_list=fit_list,
            quick_update=quick_update
        )
