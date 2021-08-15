from autolens.lens.model.visualizer import Visualizer
from autolens.imaging.plot.fit_imaging_plotters import FitImagingPlotter

from autolens.lens.model.visualizer import plot_setting


class VisualizerImaging(Visualizer):
    def visualize_fit_imaging(self, fit, during_analysis, subfolders="fit_imaging"):
        def should_plot(name):
            return plot_setting(section="fit", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_imaging_plotter = FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_imaging_plotter.figures_2d(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_image=should_plot("model_data"),
            residual_map=should_plot("residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
        )

        fit_imaging_plotter.figures_2d_of_planes(
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

                fit_imaging_plotter.figures_2d(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_2d_of_planes(
                    subtracted_image=True, model_image=True, plane_image=True
                )

            if should_plot("all_at_end_fits"):

                mat_plot_2d = self.mat_plot_2d_from(
                    subfolders="fit_imaging/fits", format="fits"
                )

                fit_imaging_plotter = FitImagingPlotter(
                    fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
                )

                fit_imaging_plotter.figures_2d(
                    image=True,
                    noise_map=True,
                    signal_to_noise_map=True,
                    model_image=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                )

                fit_imaging_plotter.figures_2d_of_planes(
                    subtracted_image=True, model_image=True
                )
