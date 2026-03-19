import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

from autoconf import conf

import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta
from autogalaxy.plot.abstract_plotters import _save_subplot

from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.tracer import Tracer
from autolens.lens.plot.tracer_plotters import TracerPlotter
from autolens.plot.abstract_plotters import Plotter, _to_lines

from autolens.lens import tracer_util


class FitInterferometerPlotter(Plotter):
    def __init__(
        self,
        fit: FitInterferometer,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

        self._lines_of_planes = None

    @property
    def _lensing_grid(self):
        return self.fit.grids.lp.mask.derive_grid.all_false

    @property
    def lines_of_planes(self) -> List[List]:
        if self._lines_of_planes is None:
            self._lines_of_planes = tracer_util.lines_of_planes_from(
                tracer=self.fit.tracer,
                grid=self._lensing_grid,
            )
        return self._lines_of_planes

    def _lines_for_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> Optional[List]:
        if remove_critical_caustic:
            return None
        try:
            return self.lines_of_planes[plane_index] or None
        except IndexError:
            return None

    @property
    def tracer(self) -> Tracer:
        return self.fit.tracer_linear_light_profiles_to_light_profiles

    def tracer_plotter_of_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> TracerPlotter:
        zoom = aa.Zoom2D(mask=self.fit.dataset.real_space_mask)

        grid = aa.Grid2D.from_extent(
            extent=zoom.extent_from(buffer=0), shape_native=zoom.shape_native
        )
        return TracerPlotter(
            tracer=self.tracer,
            grid=grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
        )

    def inversion_plotter_of_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> aplt.InversionPlotter:
        lines = None if remove_critical_caustic else self._lines_for_plane(plane_index)
        inversion_plotter = aplt.InversionPlotter(
            inversion=self.fit.inversion,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            lines=lines,
        )
        return inversion_plotter

    def plane_indexes_from(self, plane_index: int):
        if plane_index is None:
            return range(len(self.fit.tracer.planes))
        return [plane_index]

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        model_data: bool = False,
        residual_map_real: bool = False,
        residual_map_imag: bool = False,
        normalized_residual_map_real: bool = False,
        normalized_residual_map_imag: bool = False,
        chi_squared_map_real: bool = False,
        chi_squared_map_imag: bool = False,
        image: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_model_image: bool = False,
        dirty_residual_map: bool = False,
        dirty_normalized_residual_map: bool = False,
        dirty_chi_squared_map: bool = False,
        ax=None,
    ):
        self._fit_interferometer_meta_plotter.figures_2d(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            amplitudes_vs_uv_distances=amplitudes_vs_uv_distances,
            model_data=model_data,
            residual_map_real=residual_map_real,
            residual_map_imag=residual_map_imag,
            normalized_residual_map_real=normalized_residual_map_real,
            normalized_residual_map_imag=normalized_residual_map_imag,
            chi_squared_map_real=chi_squared_map_real,
            chi_squared_map_imag=chi_squared_map_imag,
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            dirty_residual_map=dirty_residual_map,
            dirty_normalized_residual_map=dirty_normalized_residual_map,
            dirty_chi_squared_map=dirty_chi_squared_map,
        )

        if image:
            plane_index = len(self.tracer.planes) - 1

            if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                tracer_plotter = self.tracer_plotter_of_plane(plane_index=plane_index)
                tracer_plotter.figures_2d(image=True, ax=ax)
            elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(
                    plane_index=plane_index
                )
                inversion_plotter.figures_2d(reconstructed_operated_data=True)

        if dirty_model_image:
            self._plot_array(
                array=self.fit.dirty_model_image,
                auto_filename="dirty_model_image_2d",
                title="Dirty Model Image",
                lines=_to_lines(self._lines_for_plane(plane_index=0)),
                ax=ax,
            )

    def figures_2d_of_planes(
        self,
        plane_index: Optional[int] = None,
        plane_image: bool = False,
        plane_noise_map: bool = False,
        plane_signal_to_noise_map: bool = False,
        zoom_to_brightest: bool = True,
        ax=None,
    ):
        if plane_image:
            if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                tracer_plotter = self.tracer_plotter_of_plane(plane_index=plane_index)
                tracer_plotter.figures_2d_of_planes(
                    plane_image=True,
                    plane_index=plane_index,
                    zoom_to_brightest=zoom_to_brightest,
                    ax=ax,
                )
            elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)
                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0,
                    reconstruction=True,
                    zoom_to_brightest=zoom_to_brightest,
                )

        if plane_noise_map:
            if self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(
                    plane_index=plane_index
                )
                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0,
                    reconstruction_noise_map=True,
                    zoom_to_brightest=zoom_to_brightest,
                )

        if plane_signal_to_noise_map:
            if self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                inversion_plotter = self.inversion_plotter_of_plane(
                    plane_index=plane_index
                )
                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=0,
                    signal_to_noise_map=True,
                    zoom_to_brightest=zoom_to_brightest,
                )

    def subplot_fit(self):
        final_plane_index = len(self.fit.tracer.planes) - 1

        fig, axes = plt.subplots(3, 4, figsize=(28, 21))
        axes = axes.flatten()

        # UV distances plot (index 0)
        self._fit_interferometer_meta_plotter._plot_yx(
            np.real(self.fit.residual_map),
            self.fit.dataset.uv_distances / 10**3.0,
            "amplitudes_vs_uv_distances",
            "Amplitudes vs UV-Distance",
            xlabel="k$\\lambda$",
            plot_axis_type="scatter",
            ax=axes[0],
        )

        self._fit_interferometer_meta_plotter._plot_array(
            self.fit.dirty_image, "dirty_image", "Dirty Image", ax=axes[1]
        )
        self._fit_interferometer_meta_plotter._plot_array(
            self.fit.dirty_signal_to_noise_map, "dirty_signal_to_noise_map", "Dirty Signal-To-Noise Map", ax=axes[2]
        )
        self._fit_interferometer_meta_plotter._plot_array(
            self.fit.dirty_model_image, "dirty_model_image_2d", "Dirty Model Image", ax=axes[3]
        )

        # source image (index 4)
        if not self.tracer.planes[final_plane_index].has(cls=aa.Pixelization):
            tracer_plotter = self.tracer_plotter_of_plane(plane_index=final_plane_index)
            tracer_plotter.figures_2d(image=True, ax=axes[4])
        else:
            inversion_plotter = self.inversion_plotter_of_plane(plane_index=final_plane_index)
            inversion_plotter.figures_2d(reconstructed_operated_data=True)

        self._fit_interferometer_meta_plotter._plot_yx(
            np.real(self.fit.normalized_residual_map),
            self.fit.dataset.uv_distances / 10**3.0,
            "real_normalized_residual_map_vs_uv_distances",
            "Norm Residual vs UV-Distance (real)",
            ylabel="$\\sigma$",
            xlabel="k$\\lambda$",
            plot_axis_type="scatter",
            ax=axes[5],
        )
        self._fit_interferometer_meta_plotter._plot_yx(
            np.imag(self.fit.normalized_residual_map),
            self.fit.dataset.uv_distances / 10**3.0,
            "imag_normalized_residual_map_vs_uv_distances",
            "Norm Residual vs UV-Distance (imag)",
            ylabel="$\\sigma$",
            xlabel="k$\\lambda$",
            plot_axis_type="scatter",
            ax=axes[6],
        )

        # source plane zoomed (index 7)
        self.figures_2d_of_planes(plane_index=final_plane_index, plane_image=True, ax=axes[7])

        self._fit_interferometer_meta_plotter._plot_array(
            self.fit.dirty_normalized_residual_map, "dirty_normalized_residual_map_2d", "Dirty Normalized Residual Map", ax=axes[8]
        )

        cmap_orig = self.cmap
        self.cmap.kwargs["vmin"] = -1.0
        self.cmap.kwargs["vmax"] = 1.0
        self._fit_interferometer_meta_plotter._plot_array(
            self.fit.dirty_normalized_residual_map, "dirty_normalized_residual_map_2d",
            r"Normalized Residual Map $1\sigma$", ax=axes[9]
        )
        self.cmap.kwargs.pop("vmin")
        self.cmap.kwargs.pop("vmax")

        self._fit_interferometer_meta_plotter._plot_array(
            self.fit.dirty_chi_squared_map, "dirty_chi_squared_map_2d", "Dirty Chi-Squared Map", ax=axes[10]
        )

        # source plane no zoom (index 11)
        self.figures_2d_of_planes(
            plane_index=final_plane_index, plane_image=True, zoom_to_brightest=False, ax=axes[11]
        )

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_fit")

    def subplot_mappings_of_plane(
        self, plane_index: Optional[int] = None, auto_filename: str = "subplot_mappings"
    ):
        if self.fit.inversion is None:
            return

        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:
            pixelization_index = 0

            inversion_plotter = self.inversion_plotter_of_plane(plane_index=0)

            fig, axes = plt.subplots(1, 4, figsize=(28, 7))
            axes = axes.flatten()

            self._fit_interferometer_meta_plotter._plot_array(
                self.fit.dirty_image, "dirty_image", "Dirty Image", ax=axes[0]
            )

            total_pixels = conf.instance["visualize"]["general"]["inversion"][
                "total_mappings_pixels"
            ]

            pix_indexes = inversion_plotter.inversion.max_pixel_list_from(
                total_pixels=total_pixels, filter_neighbors=True
            )

            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=pixelization_index, reconstructed_operated_data=True
            )

            self.figures_2d_of_planes(
                plane_index=plane_index,
                plane_image=True,
                ax=axes[2],
            )

            self.figures_2d_of_planes(
                plane_index=plane_index,
                plane_image=True,
                zoom_to_brightest=False,
                ax=axes[3],
            )

            plt.tight_layout()
            _save_subplot(fig, self.output, f"{auto_filename}_{pixelization_index}")

    def subplot_fit_real_space(self):
        if self.fit.inversion is None:
            tracer_plotter = self.tracer_plotter_of_plane(plane_index=0)
            tracer_plotter.subplot(
                image=True, source_plane=True, auto_filename="subplot_fit_real_space"
            )
        elif self.fit.inversion is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes = axes.flatten()

            inversion_plotter = self.inversion_plotter_of_plane(plane_index=1)
            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0, reconstructed_operated_data=True
            )
            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0, reconstruction=True
            )

            plt.tight_layout()
            _save_subplot(fig, self.output, "subplot_fit_real_space")
