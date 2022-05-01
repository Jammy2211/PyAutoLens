import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from os import path
import os

import autoarray as aa
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

        if should_plot("subplot_ray_tracing"):

            tracer_plotter.subplot_tracer()

        if should_plot("subplot_plane_images"):

            tracer_plotter.subplot_plane_images()

        tracer_plotter.figures_2d(
            image=should_plot("image"),
            source_plane=should_plot("source_plane_image"),
            convergence=should_plot("convergence"),
            potential=should_plot("potential"),
            deflections_y=should_plot("deflections"),
            deflections_x=should_plot("deflections"),
            magnification=should_plot("magnification"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                tracer_plotter.figures_2d(
                    image=True,
                    source_plane=True,
                    convergence=True,
                    potential=True,
                    deflections_y=True,
                    deflections_x=True,
                    magnification=True,
                )

            if should_plot("all_at_end_fits"):

                fits_mat_plot_2d = self.mat_plot_2d_from(
                    subfolders=path.join("ray_tracing", "fits"), format="fits"
                )

                tracer_plotter = TracerPlotter(
                    tracer=tracer,
                    grid=grid,
                    mat_plot_2d=fits_mat_plot_2d,
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

    def visualize_hyper_images(
        self,
        hyper_galaxy_image_path_dict: {str, aa.Array2D},
        hyper_model_image: aa.Array2D,
    ):
        """
        Visualizes the hyper-images and hyper dataset inferred by a model-fit.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `hyper`. When
        used with a non-linear search the `visualize_path` points to the search's results folder.

        Visualization includes individual images of attributes of the hyper dataset (e.g. the hyper model image) and
        a subplot of all hyper galaxy images on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [hyper] header.

        Parameters
        ----------
        hyper_galaxy_image_path_dict
            A dictionary mapping the path to each galaxy (e.g. its name) to its corresponding hyper galaxy image.
        hyper_model_image
            The hyper model image which corresponds to the sum of hyper galaxy images.
        """

        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = aplt.HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            hyper_plotter.figure_hyper_model_image(hyper_model_image=hyper_model_image)

        if should_plot("images_of_galaxies"):

            hyper_plotter.subplot_hyper_images_of_galaxies(
                hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict
            )

    def visualize_contribution_maps(self, tracer: Tracer):
        """
        Visualizes the contribution maps that are used for hyper features which adapt a model to the dataset it is
        fitting.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `hyper`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood contribution maps inferred by the search so far.

        Visualization includes individual images of attributes of the hyper dataset (e.g. the contribution map of
        each galaxy) and a subplot of all contribution maps on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [hyper] header.

        Parameters
        ----------
        tracer
            The maximum log likelihood `Tracer` of the non-linear search which is used to plot the contribution maps.
        """

        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = aplt.HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if hasattr(tracer, "contribution_map_list"):
            if should_plot("contribution_map_list"):
                hyper_plotter.subplot_contribution_map_list(
                    contribution_map_list_list=tracer.contribution_map_list
                )

    def visualize_stochastic_histogram(
        self,
        stochastic_log_likelihoods: np.ndarray,
        max_log_evidence: float,
        histogram_bins: int = 10,
    ):
        """
        Certain `Inversion`'s have stochasticity in their log likelihood estimate.

        For example, the `VoronoiBrightnessImage` pixelization, which changes the likelihood depending on how different
        KMeans seeds change the pixel-grid.

        A log likelihood cap can be applied to model-fits performed using these `Inversion`'s to improve error and
        posterior estimates. This log likelihood cap is estimated from a list of stochastic log likelihoods, where
        these log likelihoods are computed using the same model but with different KMeans seeds.

        This function plots a histogram representing the distribution of these stochastic log likelihoods with a 1D
        Gaussian fit to the likelihoods overlaid. This figure can be used to determine how subject the fit to this
        dataset is to the stochastic likelihood effect.

        Parameters
        ----------
        stochastic_log_likelihoods
            The stochastic log likelihood which are used to plot the histogram and Gaussian.
        max_log_evidence
            The maximum log likelihood value of the non-linear search, which will likely be much larger than any of the
            stochastic log likelihoods as it will be boosted high relative to most samples.
        histogram_bins
            The number of bins in the histogram used to visualize the distribution of stochastic log likelihoods.

        Returns
        -------
        float
            A log likelihood cap which is applied in a stochastic model-fit to give improved error and posterior
            estimates.
        """
        if stochastic_log_likelihoods is None:
            return

        if plot_setting("other", "stochastic_histogram"):

            file_path = path.join(self.visualize_path, "other")

            try:
                os.makedirs(file_path)
            except FileExistsError or IsADirectoryError:
                pass

            filename = path.join(file_path, "stochastic_histogram.png")

            if path.exists(filename):
                try:
                    os.rmdir(filename)
                except Exception:
                    pass

            (mu, sigma) = norm.fit(stochastic_log_likelihoods)
            n, bins, patches = plt.hist(
                x=stochastic_log_likelihoods, bins=histogram_bins, density=1
            )
            y = norm.pdf(bins, mu, sigma)
            plt.plot(bins, y, "--")
            plt.xlabel("log evidence")
            plt.title("Stochastic Log Evidence Histogram")
            plt.axvline(max_log_evidence, color="r")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
