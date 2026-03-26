import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.plotter import plot_setting

from autogalaxy.analysis.plotter import Plotter as AgPlotter

from autolens.lens.tracer import Tracer
from autolens.lens.plot.tracer_plots import (
    subplot_tracer,
    subplot_galaxies_images,
    save_tracer_fits,
    save_source_plane_images_fits,
)
from autoarray.plot.array import plot_array


class Plotter(AgPlotter):
    """
    Visualizes the maximum log likelihood model of a model-fit, including components of the model and fit objects.

    The methods of the `Plotter` are called throughout a non-linear search using the `Analysis`
    classes `visualize` method.

    The images output by the `Plotter` are customized using the file `config/visualize/plots.yaml`.

    Parameters
    ----------
    image_path
        The path on the hard-disk to the `image` folder of the non-linear searches results.
    """

    def tracer(
        self,
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
    ):
        """
        Visualizes a `Tracer` object.

        Parameters
        ----------
        tracer
            The maximum log likelihood `Tracer` of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing.
        """

        def should_plot(name):
            return plot_setting(section="tracer", name=name)

        output_path = str(self.image_path)
        fmt = self.fmt

        if should_plot("subplot_tracer"):
            subplot_tracer(
                tracer=tracer,
                grid=grid,
                output_path=output_path,
                output_format=fmt,
            )

        if should_plot("subplot_galaxies_images"):
            subplot_galaxies_images(
                tracer=tracer,
                grid=grid,
                output_path=output_path,
                output_format=fmt,
            )

        if should_plot("fits_tracer"):
            save_tracer_fits(tracer=tracer, grid=grid, output_path=self.image_path)

        if should_plot("fits_source_plane_images"):
            save_source_plane_images_fits(tracer=tracer, grid=grid, output_path=self.image_path)

    def image_with_positions(self, image: aa.Array2D, positions: aa.Grid2DIrregular):
        """
        Visualizes the positions of a model-fit.

        Parameters
        ----------
        image
            The imaging dataset whose image the positions are overlaid.
        positions
            The 2D (y,x) arc-second positions used to penalize inaccurate mass models.
        """

        def should_plot(name):
            return plot_setting(section=["positions"], name=name)

        if positions is not None and should_plot("image_with_positions"):
            pos_arr = np.array(
                positions.array if hasattr(positions, "array") else positions
            )

            fmt = self.fmt
            if isinstance(fmt, (list, tuple)):
                fmt = fmt[0]

            plot_array(
                array=image,
                positions=[pos_arr],
                title="Image With Positions",
                output_path=str(self.image_path),
                output_filename="image_with_positions",
                output_format=fmt,
            )
