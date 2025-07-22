import autofit as af
import autogalaxy as ag

from autolens.interferometer.model.plotter_interface import (
    PlotterInterfaceInterferometer,
)
from autolens.lens import tracer_util
from autogalaxy import exc


class VisualizerInterferometer(af.Visualizer):
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

        plotter_interface = PlotterInterfaceInterferometer(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter_interface.interferometer(dataset=analysis.interferometer)

        if analysis.positions_likelihood_list is not None:

            positions_list = []

            for positions_likelihood in analysis.positions_likelihood_list:
                positions_list.append(positions_likelihood.positions)

            positions = ag.Grid2DIrregular(positions_list)

            plotter_interface.image_with_positions(
                image=analysis.dataset.dirty_image, positions=positions
            )

        if analysis.adapt_images is not None:
            plotter_interface.adapt_images(adapt_images=analysis.adapt_images)

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Outputs images of the maximum log likelihood model inferred by the model-fit. This function is called
        throughout the non-linear search at input intervals, and therefore provides on-the-fly visualization of how
        well the model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Tracer`, including the images of each of its galaxies.

        - Images of the best-fit `FitInterferometer`, including the model-image, residuals and chi-squared of its fit
          to the imaging data.

        - The adapt-images of the model-fit showing how the galaxies are used to represent different galaxies in
          the dataset.

        - If adapt features are used to scale the noise, a `FitInterferometer` with these features turned off may be
          output, to indicate how much these features are altering the dataset.

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

        if analysis.positions_likelihood_list is not None:

            overwrite_file = True

            for positions_likelihood in analysis.positions_likelihood_list:

                positions_likelihood.output_positions_info(
                    output_path=paths.output_path,
                    tracer=fit.tracer,
                    overwrite_file=overwrite_file,
                )

                overwrite_file = False

        if fit.inversion is not None:
            try:
                fit.inversion.reconstruction
            except exc.InversionException:
                return

        visuals_2d_of_planes_list = tracer_util.visuals_2d_of_planes_list_from(
            tracer=fit.tracer, grid=fit.grids.lp.mask.derive_grid.all_false
        )

        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        zoom = ag.Zoom2D(mask=fit.dataset.real_space_mask)

        extent = zoom.extent_from(buffer=0)
        shape_native = zoom.shape_native

        grid = ag.Grid2D.from_extent(extent=extent, shape_native=shape_native)

        plotter_interface = PlotterInterfaceInterferometer(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        try:
            plotter_interface.fit_interferometer(
                fit=fit, visuals_2d_of_planes_list=visuals_2d_of_planes_list
            )
        except exc.InversionException:
            pass

        plotter_interface.tracer(
            tracer=tracer,
            grid=grid,
            visuals_2d_of_planes_list=visuals_2d_of_planes_list,
        )
        plotter_interface.galaxies(
            galaxies=tracer.galaxies,
            grid=fit.grids.lp,
        )
        if fit.inversion is not None:
            try:
                plotter_interface.inversion(
                    inversion=fit.inversion,
                )
            except IndexError:
                pass
