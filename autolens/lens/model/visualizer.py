from scipy.stats import norm
import matplotlib.pyplot as plt
from os import path
import os

from autoconf import conf
import autogalaxy.plot as aplt

from autogalaxy.analysis.visualizer import Visualizer as AgVisualizer

from autolens.lens.plot.ray_tracing_plotters import TracerPlotter


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer(AgVisualizer):
    def visualize_tracer(self, tracer, grid, during_analysis):
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

    def visualize_hyper_images(self, hyper_galaxy_image_path_dict, hyper_model_image):
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

    def visualize_contribution_maps(self, tracer):
        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = aplt.HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if hasattr(tracer, "contribution_maps_of_galaxies"):
            if should_plot("contribution_maps_of_galaxies"):
                hyper_plotter.subplot_contribution_maps_of_galaxies(
                    contribution_maps_of_galaxies=tracer.contribution_maps_of_galaxies
                )

    def visualize_stochastic_histogram(
        self, log_evidences, max_log_evidence, histogram_bins=10
    ):

        if log_evidences is None:
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

            (mu, sigma) = norm.fit(log_evidences)
            n, bins, patches = plt.hist(x=log_evidences, bins=histogram_bins, density=1)
            y = norm.pdf(bins, mu, sigma)
            plt.plot(bins, y, "--")
            plt.xlabel("log evidence")
            plt.title("Stochastic Log Evidence Histogram")
            plt.axvline(max_log_evidence, color="r")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
