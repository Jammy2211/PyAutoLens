from os import path

import autoarray as aa
import autogalaxy.plot as aplt

from autogalaxy.analysis.plotter_interface import plot_setting

from autogalaxy.analysis.plotter_interface import PlotterInterface as AgPlotterInterface

from autolens.lens.tracer import Tracer
from autolens.lens.plot.tracer_plotters import TracerPlotter


class PlotterInterface(AgPlotterInterface):
    """
    Visualizes the maximum log likelihood model of a model-fit, including components of the model and fit objects.

    The methods of the `PlotterInterface` are called throughout a non-linear search using the `Analysis`
    classes `visualize` method.

    The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml`.

    Parameters
    ----------
    image_path
        The path on the hard-disk to the `image` folder of the non-linear searches results.
    """

    def tracer(self, tracer: Tracer, grid: aa.type.Grid2DLike, during_analysis: bool):
        """
        Visualizes a `Tracer` object.

        Images are output to the `image` folder of the `image_path` in a subfolder called `tracer`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Tracer` inferred by the search so far.

        Visualization includes individual images of attributes of the tracer (e.g. its image, convergence, deflection
        angles) and a subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [tracer] header.

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
            return plot_setting(section="tracer", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="tracer")

        tracer_plotter = TracerPlotter(
            tracer=tracer,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_galaxies_images"):
            tracer_plotter.subplot_galaxies_images()

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
                subfolders=path.join("tracer", "end"),
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
                subfolders=path.join("tracer", "fits"), format="fits"
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

    def image_with_positions(self, image: aa.Array2D, positions: aa.Grid2DIrregular):
        """
        Visualizes the positions of a model-fit, where these positions are used to resample lens models where
        the positions to do trace within an input threshold of one another in the source-plane.

        Images are output to the `image` folder of the `image_path` in a subfolder called `positions`. When
        used with a non-linear search the `image_path` points to the search's results folder.

        The visualization is an image of the strong lens with the positions overlaid.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [tracer] header.

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
