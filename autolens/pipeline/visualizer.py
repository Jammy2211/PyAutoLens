from scipy.stats import norm
import matplotlib.pyplot as plt
from os import path
import os

from autoconf import conf
from autogalaxy.plot.plotters import hyper_plotters
from autogalaxy.pipeline import visualizer
from autolens.plot.plotters import (
    fit_interferometer_plotters,
    ray_tracing_plotters,
    fit_imaging_plotters,
)


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer(visualizer.Visualizer):
    def visualize_tracer(self, tracer, grid, during_analysis):
        def should_plot(name):
            return plot_setting(section="ray_tracing", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="ray_tracing")

        tracer_plotter = ray_tracing_plotters.TracerPlotter(
            tracer=tracer,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_ray_tracing"):

            tracer_plotter.subplot_tracer()

        tracer_plotter.figures(
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

                tracer_plotter.figures(
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

                tracer_plotter = ray_tracing_plotters.TracerPlotter(
                    tracer=tracer,
                    grid=grid,
                    mat_plot_2d=fits_mat_plot_2d,
                    include_2d=self.include_2d,
                )

                tracer_plotter.figures(
                    image=True,
                    source_plane=True,
                    convergence=True,
                    potential=True,
                    deflections_y=True,
                    deflections_x=True,
                    magnification=True,
                )

    def visualize_fit_imaging(self, fit, during_analysis, subfolders="fit_imaging"):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_imaging_plotter = fit_imaging_plotters.FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_imaging_plotter.figures(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_image=should_plot("model_data"),
            residual_map=should_plot("residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
        )

        fit_imaging_plotter.figures_of_planes(
            subtracted_image=should_plot("subtracted_images_of_planes"),
            model_image=should_plot("model_images_of_planes"),
            plane_image=should_plot("plane_images_of_planes"),
        )

        if should_plot("subplot_fit"):
            fit_imaging_plotter.subplot_fit_imaging()

        if should_plot("subplots_of_planes_fits"):
            fit_imaging_plotter.subplot_of_planes()

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_imaging_plotter.figures(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_of_planes(
                    subtracted_image=True, model_image=True, plane_image=True
                )

            if should_plot("all_at_end_fits"):

                mat_plot_2d = self.mat_plot_2d_from(
                    subfolders="fit_imaging/fits", format="fits"
                )

                fit_imaging_plotter = fit_imaging_plotters.FitImagingPlotter(
                    fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
                )

                fit_imaging_plotter.figures(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_of_planes(
                    subtracted_image=True, model_image=True
                )

    def visualize_fit_interferometer(
        self, fit, during_analysis, subfolders="fit_interferometer"
    ):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders=subfolders)
        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_interferometer_plotter = fit_interferometer_plotters.FitInterferometerPlotter(
            fit=fit,
            include_2d=self.include_2d,
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_fit"):
            fit_interferometer_plotter.subplot_fit_interferometer()
            fit_interferometer_plotter.subplot_fit_real_space()

        fit_interferometer_plotter.figures(
            visibilities=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_visibilities=should_plot("model_data"),
            residual_map_real=should_plot("residual_map"),
            chi_squared_map_real=should_plot("chi_squared_map"),
            normalized_residual_map_real=should_plot("normalized_residual_map"),
            residual_map_imag=should_plot("residual_map"),
            chi_squared_map_imag=should_plot("chi_squared_map"),
            normalized_residual_map_imag=should_plot("normalized_residual_map"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                fit_interferometer_plotter.figures(
                    visibilities=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_visibilities=True,
                    residual_map_real=True,
                    chi_squared_map_real=True,
                    normalized_residual_map_real=True,
                    residual_map_imag=True,
                    chi_squared_map_imag=True,
                    normalized_residual_map_imag=True,
                )

            if should_plot("all_at_end_fits"):

                mat_plot_2d = self.mat_plot_2d_from(
                    subfolders="fit_interferometer/fits", format="fits"
                )

                fit_interferometer_plotter = fit_interferometer_plotters.FitInterferometerPlotter(
                    fit=fit, include_2d=self.include_2d, mat_plot_2d=mat_plot_2d
                )

                fit_interferometer_plotter.figures(
                    visibilities=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_visibilities=True,
                    residual_map_real=True,
                    chi_squared_map_real=True,
                    normalized_residual_map_real=True,
                    residual_map_imag=True,
                    chi_squared_map_imag=True,
                    normalized_residual_map_imag=True,
                )

    def visualize_hyper_images(
        self, hyper_galaxy_image_path_dict, hyper_model_image, tracer
    ):
        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = hyper_plotters.HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            hyper_plotter.figure_hyper_model_image(hyper_model_image=hyper_model_image)

        if should_plot("images_of_galaxies"):

            hyper_plotter.subplot_hyper_images_of_galaxies(
                hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict
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
