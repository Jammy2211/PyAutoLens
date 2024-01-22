from os import path

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autogalaxy.analysis.visualizer import plot_setting

from autogalaxy.analysis.visualizer import Visualizer as AgVisualizer

from autolens.lens.ray_tracing import Tracer
from autolens.lens.plot.ray_tracing_plotters import TracerPlotter


class Visualizer(AgVisualizer):
    """
    Visualizes the maximum log likelihood model of a model-fit, including components of the model and fit objects.

    The methods of the `Visualizer` are called throughout a non-linear search using the `Analysis`
    classes `visualize` method.

    The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini`.

    Parameters
    ----------
    visualize_path
        The path on the hard-disk to the `image` folder of the non-linear searches results.
    """

    def visualize_tracer(
        self, tracer: Tracer, grid: aa.type.Grid2DLike, during_analysis: bool
    ):
        """
        Visualizes a `Tracer` object.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `ray_tracing`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Tracer` inferred by the search so far.

        Visualization includes individual images of attributes of the tracer (e.g. its image, convergence, deflection
        angles) and a subplot of all these attributes on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [ray_tracing] header.

        Parameters
        ----------
        tracer
            The maximum log likelihood `Tracer` of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing, which is the masked grid tied to
            the dataset.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        def should_plot(name):
            return plot_setting(section="ray_tracing", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="ray_tracing")

        tracer_plotter = TracerPlotter(
            tracer=tracer,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_plane_images"):
            tracer_plotter.subplot_plane_images()

        tracer_plotter.figures_2d(
            image=should_plot("image"),
            source_plane=should_plot("source_plane_image"),
            deflections_y=should_plot("deflections"),
            deflections_x=should_plot("deflections"),
            magnification=should_plot("magnification"),
        )

        mat_plot_2d.use_log10 = True

        tracer_plotter.figures_2d(
            convergence=should_plot("convergence"),
            potential=should_plot("potential"),
        )

        if should_plot("lens_image"):
            tracer_plotter.figures_2d_of_planes(
                plane_image=True, plane_index=0, zoom_to_brightest=False
            )

        mat_plot_2d.use_log10 = False

        if not during_analysis and should_plot("all_at_end_png"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join("ray_tracing", "end"),
            )

            tracer_plotter = TracerPlotter(
                tracer=tracer,
                grid=grid,
                mat_plot_2d=mat_plot_2d,
                include_2d=self.include_2d,
            )

            tracer_plotter.figures_2d(
                image=True,
                source_plane=True,
                deflections_y=True,
                deflections_x=True,
                magnification=True,
            )

            mat_plot_2d.use_log10 = True

            tracer_plotter.figures_2d(
                convergence=True,
                potential=True,
            )

            tracer_plotter.figures_2d_of_planes(
                plane_image=True, plane_index=0, zoom_to_brightest=False
            )

            mat_plot_2d.use_log10 = False

        if not during_analysis and should_plot("all_at_end_fits"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join("ray_tracing", "fits"), format="fits"
            )

            tracer_plotter = TracerPlotter(
                tracer=tracer,
                grid=grid,
                mat_plot_2d=mat_plot_2d,
                include_2d=self.include_2d,
            )

            tracer_plotter.figures_2d(
                image=True,
                source_plane=True,
                convergence=True,
                potential=True,
                deflections_y=True,
                deflections_x=True,
                magnification=True,
            )

    def visualize_image_with_positions(
        self, image: aa.Array2D, positions: aa.Grid2DIrregular
    ):
        """
        Visualizes the positions of a model-fit, where these positions are used to resample lens models where
        the positions to do trace within an input threshold of one another in the source-plane.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `positions`. When
        used with a non-linear search the `visualize_path` points to the search's results folder.

        The visualization is an image of the strong lens with the positions overlaid.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [ray_tracing] header.

        Parameters
        ----------
        imaging
            The imaging dataset whose image the positions are overlaid.
        position
            The 2D (y,x) arc-second positions used to resample inaccurate mass models.
        """

        def should_plot(name):
            return plot_setting(section=["positions"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="positions")

        if positions is not None:
            visuals_2d = aplt.Visuals2D(positions=positions)

            image_plotter = aplt.Array2DPlotter(
                array=image,
                mat_plot_2d=mat_plot_2d,
                include_2d=self.include_2d,
                visuals_2d=visuals_2d,
            )
            image_plotter.set_filename("image_with_positions")
            if should_plot("image_with_positions"):
                image_plotter.figure_2d()

    def visualize_adapt_images(self, adapt_images: ag.AdaptImages):
        """
        Visualizes the adapt-images and adapt image inferred by a model-fit.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `adapt`. When
        used with a non-linear search the `visualize_path` points to the search's results folder.

        Visualization includes individual images of attributes of the adapt image (e.g. the adapt image) and
        a subplot of all galaxy images on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [adapt] header.

        Parameters
        ----------
        adapt_images
            The adapt images (e.g. overall model image, individual galaxy images).
        """

        def should_plot(name):
            return plot_setting(section="adapt", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="adapt")

        adapt_plotter = aplt.AdaptPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            adapt_plotter.figure_model_image(model_image=adapt_images.model_image)

        if should_plot("images_of_galaxies"):
            adapt_plotter.subplot_images_of_galaxies(
                adapt_galaxy_name_image_dict=adapt_images.galaxy_image_dict
            )

    def visualize_contribution_maps(self, tracer: Tracer):
        """
        Visualizes the contribution maps that are used for adapt features which adapt a model to the dataset it is
        fitting.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `adapt`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood contribution maps inferred by the search so far.

        Visualization includes individual images of attributes of the adapt image (e.g. the contribution map of
        each galaxy) and a subplot of all contribution maps on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [adapt] header.

        Parameters
        ----------
        tracer
            The maximum log likelihood `Tracer` of the non-linear search which is used to plot the contribution maps.
        """

        def should_plot(name):
            return plot_setting(section="adapt", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="adapt")

        adapt_plotter = aplt.AdaptPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if hasattr(tracer, "contribution_map_list"):
            if should_plot("contribution_map_list"):
                adapt_plotter.subplot_contribution_map_list(
                    contribution_map_list_list=tracer.contribution_map_list
                )
