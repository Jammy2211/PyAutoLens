import copy
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

from autoconf import conf

import autoarray as aa
import autogalaxy.plot as aplt

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta
from autogalaxy.plot.abstract_plotters import _save_subplot

from autolens.plot.abstract_plotters import Plotter, _to_lines
from autolens.imaging.fit_imaging import FitImaging
from autolens.lens.plot.tracer_plotters import TracerPlotter

from autolens.lens import tracer_util


class FitImagingPlotter(Plotter):
    def __init__(
        self,
        fit: FitImaging,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.residuals_symmetric_cmap = residuals_symmetric_cmap
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
    def tracer(self):
        return self.fit.tracer_linear_light_profiles_to_light_profiles

    def tracer_plotter_of_plane(
        self, plane_index: int, remove_critical_caustic: bool = False
    ) -> TracerPlotter:
        zoom = aa.Zoom2D(mask=self.fit.mask)

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

    def figures_2d_of_planes(
        self,
        plane_index: Optional[int] = None,
        subtracted_image: bool = False,
        model_image: bool = False,
        plane_image: bool = False,
        plane_noise_map: bool = False,
        plane_signal_to_noise_map: bool = False,
        use_source_vmax: bool = False,
        zoom_to_brightest: bool = True,
        remove_critical_caustic: bool = False,
        ax=None,
    ):
        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:

            if use_source_vmax:
                self.cmap.kwargs["vmax"] = np.max(
                    self.fit.model_images_of_planes_list[plane_index].array
                )

            if subtracted_image:

                title = f"Subtracted Image of Plane {plane_index}"
                filename = f"subtracted_image_of_plane_{plane_index}"

                if len(self.tracer.planes) == 2:
                    if plane_index == 0:
                        title = "Source Subtracted Image"
                        filename = "source_subtracted_image"
                    elif plane_index == 1:
                        title = "Lens Subtracted Image"
                        filename = "lens_subtracted_image"

                self._plot_array(
                    array=self.fit.subtracted_images_of_planes_list[plane_index],
                    auto_filename=filename,
                    title=title,
                    lines=_to_lines(
                        self._lines_for_plane(
                            plane_index=plane_index,
                            remove_critical_caustic=remove_critical_caustic,
                        )
                    ),
                    ax=ax,
                )

            if model_image:

                title = f"Model Image of Plane {plane_index}"
                filename = f"model_image_of_plane_{plane_index}"

                if len(self.tracer.planes) == 2:
                    if plane_index == 0:
                        title = "Lens Model Image"
                        filename = "lens_model_image"
                    elif plane_index == 1:
                        title = "Source Model Image"
                        filename = "source_model_image"

                self._plot_array(
                    array=self.fit.model_images_of_planes_list[plane_index],
                    auto_filename=filename,
                    title=title,
                    lines=_to_lines(
                        self._lines_for_plane(
                            plane_index=plane_index,
                            remove_critical_caustic=remove_critical_caustic,
                        )
                    ),
                    ax=ax,
                )

            if plane_image:

                if not self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    tracer_plotter = self.tracer_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )

                    tracer_plotter.figures_2d_of_planes(
                        plane_image=True,
                        plane_index=plane_index,
                        zoom_to_brightest=zoom_to_brightest,
                        ax=ax,
                    )

                elif self.tracer.planes[plane_index].has(cls=aa.Pixelization):

                    inversion_plotter = self.inversion_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )

                    inversion_plotter.figures_2d_of_pixelization(
                        pixelization_index=0,
                        reconstruction=True,
                        zoom_to_brightest=zoom_to_brightest,
                    )

            if use_source_vmax:
                try:
                    self.cmap.kwargs.pop("vmax")
                except KeyError:
                    pass

            if plane_noise_map:
                if self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                    inversion_plotter = self.inversion_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )
                    inversion_plotter.figures_2d_of_pixelization(
                        pixelization_index=0,
                        reconstruction_noise_map=True,
                        zoom_to_brightest=zoom_to_brightest,
                    )

            if plane_signal_to_noise_map:
                if self.tracer.planes[plane_index].has(cls=aa.Pixelization):
                    inversion_plotter = self.inversion_plotter_of_plane(
                        plane_index=plane_index,
                        remove_critical_caustic=remove_critical_caustic,
                    )
                    inversion_plotter.figures_2d_of_pixelization(
                        pixelization_index=0,
                        signal_to_noise_map=True,
                        zoom_to_brightest=zoom_to_brightest,
                    )

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        residual_flux_fraction_map: bool = False,
        use_source_vmax: bool = False,
        suffix: str = "",
        ax=None,
    ):
        if use_source_vmax:
            try:
                source_vmax = np.max(
                    [
                        model_image_plane.array
                        for model_image_plane in self.fit.model_images_of_planes_list[1:]
                    ]
                )
            except ValueError:
                source_vmax = None
        else:
            source_vmax = None

        if data:
            if use_source_vmax and source_vmax is not None:
                self.cmap.kwargs["vmax"] = source_vmax

            self._plot_array(
                array=self.fit.data,
                auto_filename=f"data{suffix}",
                title="Data",
                ax=ax,
            )

            if use_source_vmax and source_vmax is not None:
                self.cmap.kwargs.pop("vmax")

        if noise_map:
            self._plot_array(
                array=self.fit.noise_map,
                auto_filename=f"noise_map{suffix}",
                title="Noise-Map",
                ax=ax,
            )

        if signal_to_noise_map:
            self._plot_array(
                array=self.fit.signal_to_noise_map,
                auto_filename=f"signal_to_noise_map{suffix}",
                title="Signal-To-Noise Map",
                ax=ax,
            )

        if model_image:
            if use_source_vmax and source_vmax is not None:
                self.cmap.kwargs["vmax"] = source_vmax

            self._plot_array(
                array=self.fit.model_data,
                auto_filename=f"model_image{suffix}",
                title="Model Image",
                lines=_to_lines(self._lines_for_plane(plane_index=0)),
                ax=ax,
            )

            if use_source_vmax and source_vmax is not None:
                self.cmap.kwargs.pop("vmax")

        cmap_original = self.cmap

        if self.residuals_symmetric_cmap:
            self.cmap = self.cmap.symmetric_cmap_from()

        if residual_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_filename=f"residual_map{suffix}",
                title="Residual Map",
                ax=ax,
            )

        if normalized_residual_map:
            self._plot_array(
                array=self.fit.normalized_residual_map,
                auto_filename=f"normalized_residual_map{suffix}",
                title="Normalized Residual Map",
                ax=ax,
            )

        self.cmap = cmap_original

        if chi_squared_map:
            self._plot_array(
                array=self.fit.chi_squared_map,
                auto_filename=f"chi_squared_map{suffix}",
                title="Chi-Squared Map",
                ax=ax,
            )

        if residual_flux_fraction_map:
            self._plot_array(
                array=self.fit.residual_flux_fraction_map,
                auto_filename=f"residual_flux_fraction_map{suffix}",
                title="Residual Flux Fraction Map",
                ax=ax,
            )

    def subplot_fit_x1_plane(self):
        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        axes = axes.flatten()

        self.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data", ax=axes[0])
        self.cmap.kwargs.pop("vmax")

        self._fit_imaging_meta_plotter._plot_array(
            self.fit.signal_to_noise_map, "signal_to_noise_map", "Signal-To-Noise Map", ax=axes[1]
        )

        self.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self._fit_imaging_meta_plotter._plot_array(self.fit.model_data, "model_image", "Model Image", ax=axes[2])
        self.cmap.kwargs.pop("vmax")

        self.residuals_symmetric_cmap = False
        cmap_orig = self.cmap
        norm_resid = self.fit.normalized_residual_map
        self._fit_imaging_meta_plotter._plot_array(norm_resid, "normalized_residual_map", "Lens Light Subtracted", ax=axes[3])

        self.cmap.kwargs["vmin"] = 0.0
        self._fit_imaging_meta_plotter._plot_array(norm_resid, "normalized_residual_map", "Subtracted Image Zero Minimum", ax=axes[4])
        self.cmap.kwargs.pop("vmin")

        self.residuals_symmetric_cmap = True
        self.cmap = cmap_orig.symmetric_cmap_from()
        self._fit_imaging_meta_plotter._plot_array(norm_resid, "normalized_residual_map", "Normalized Residual Map", ax=axes[5])
        self.cmap = cmap_orig

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_fit_x1_plane")

    def subplot_fit_log10_x1_plane(self):
        use_log10_orig = self.use_log10
        self.use_log10 = True

        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        axes = axes.flatten()

        self.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data", ax=axes[0])
        self.cmap.kwargs.pop("vmax")

        self._fit_imaging_meta_plotter._plot_array(
            self.fit.signal_to_noise_map, "signal_to_noise_map", "Signal-To-Noise Map", ax=axes[1]
        )

        self.cmap.kwargs["vmax"] = np.max(self.fit.model_images_of_planes_list[0].array)
        self._fit_imaging_meta_plotter._plot_array(self.fit.model_data, "model_image", "Model Image", ax=axes[2])
        self.cmap.kwargs.pop("vmax")

        self.residuals_symmetric_cmap = False
        norm_resid = self.fit.normalized_residual_map
        self._fit_imaging_meta_plotter._plot_array(norm_resid, "normalized_residual_map", "Lens Light Subtracted", ax=axes[3])

        self.residuals_symmetric_cmap = True
        cmap_sym = self.cmap.symmetric_cmap_from()
        self._fit_imaging_meta_plotter._plot_array(norm_resid, "normalized_residual_map", "Normalized Residual Map", ax=axes[4])

        self._fit_imaging_meta_plotter._plot_array(self.fit.chi_squared_map, "chi_squared_map", "Chi-Squared Map", ax=axes[5])

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_fit_log10")

        self.use_log10 = use_log10_orig
        self.residuals_symmetric_cmap = True

    def subplot_fit(self, plane_index: Optional[int] = None):
        if len(self.fit.tracer.planes) == 1:
            return self.subplot_fit_x1_plane()

        plane_index_tag = "" if plane_index is None else f"_{plane_index}"

        final_plane_index = (
            len(self.fit.tracer.planes) - 1 if plane_index is None else plane_index
        )

        try:
            source_vmax = np.max(
                [mi.array for mi in self.fit.model_images_of_planes_list[1:]]
            )
        except ValueError:
            source_vmax = None

        fig, axes = plt.subplots(3, 4, figsize=(28, 21))
        axes = axes.flatten()

        self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data", ax=axes[0])

        if source_vmax is not None:
            self.cmap.kwargs["vmax"] = source_vmax
        self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data (Source Scale)", ax=axes[1])
        if source_vmax is not None:
            self.cmap.kwargs.pop("vmax")

        self._fit_imaging_meta_plotter._plot_array(
            self.fit.signal_to_noise_map, "signal_to_noise_map", "Signal-To-Noise Map", ax=axes[2]
        )
        self._fit_imaging_meta_plotter._plot_array(self.fit.model_data, "model_image", "Model Image", ax=axes[3])

        lens_model_img = self.fit.model_images_of_planes_list[0]
        self._fit_imaging_meta_plotter._plot_array(lens_model_img, "lens_model_image", "Lens Light Model Image", ax=axes[4])

        if source_vmax is not None:
            self.cmap.kwargs["vmin"] = 0.0
            self.cmap.kwargs["vmax"] = source_vmax

        subtracted_img = self.fit.subtracted_images_of_planes_list[final_plane_index]
        self._fit_imaging_meta_plotter._plot_array(subtracted_img, "subtracted_image", "Lens Light Subtracted", ax=axes[5])

        source_model_img = self.fit.model_images_of_planes_list[final_plane_index]
        self._fit_imaging_meta_plotter._plot_array(source_model_img, "source_model_image", "Source Model Image", ax=axes[6])

        if source_vmax is not None:
            self.cmap.kwargs.pop("vmin")
            self.cmap.kwargs.pop("vmax")

        self.figures_2d_of_planes(
            plane_index=final_plane_index, plane_image=True, use_source_vmax=True, ax=axes[7]
        )

        cmap_orig = self.cmap
        if self.residuals_symmetric_cmap:
            self.cmap = self.cmap.symmetric_cmap_from()
        self._fit_imaging_meta_plotter._plot_array(
            self.fit.normalized_residual_map, "normalized_residual_map", "Normalized Residual Map", ax=axes[8]
        )

        self.cmap.kwargs["vmin"] = -1.0
        self.cmap.kwargs["vmax"] = 1.0
        self._fit_imaging_meta_plotter._plot_array(
            self.fit.normalized_residual_map, "normalized_residual_map", r"Normalized Residual Map $1\sigma$", ax=axes[9]
        )
        self.cmap.kwargs.pop("vmin")
        self.cmap.kwargs.pop("vmax")
        self.cmap = cmap_orig

        self._fit_imaging_meta_plotter._plot_array(self.fit.chi_squared_map, "chi_squared_map", "Chi-Squared Map", ax=axes[10])

        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True,
            ax=axes[11],
        )

        plt.tight_layout()
        _save_subplot(fig, self.output, f"subplot_fit{plane_index_tag}")

    def subplot_fit_log10(self, plane_index: Optional[int] = None):
        if len(self.fit.tracer.planes) == 1:
            return self.subplot_fit_log10_x1_plane()

        use_log10_orig = self.use_log10
        self.use_log10 = True

        plane_index_tag = "" if plane_index is None else f"_{plane_index}"
        final_plane_index = (
            len(self.fit.tracer.planes) - 1 if plane_index is None else plane_index
        )

        try:
            source_vmax = np.max(
                [mi.array for mi in self.fit.model_images_of_planes_list[1:]]
            )
        except ValueError:
            source_vmax = None

        fig, axes = plt.subplots(3, 4, figsize=(28, 21))
        axes = axes.flatten()

        self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data", ax=axes[0])

        if source_vmax is not None:
            self.cmap.kwargs["vmax"] = source_vmax
        try:
            self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data (Source Scale)", ax=axes[1])
        except ValueError:
            pass
        if source_vmax is not None:
            self.cmap.kwargs.pop("vmax", None)

        try:
            self._fit_imaging_meta_plotter._plot_array(
                self.fit.signal_to_noise_map, "signal_to_noise_map", "Signal-To-Noise Map", ax=axes[2]
            )
        except ValueError:
            pass

        self._fit_imaging_meta_plotter._plot_array(self.fit.model_data, "model_image", "Model Image", ax=axes[3])

        lens_model_img = self.fit.model_images_of_planes_list[0]
        self._fit_imaging_meta_plotter._plot_array(lens_model_img, "lens_model_image", "Lens Light Model Image", ax=axes[4])

        if source_vmax is not None:
            self.cmap.kwargs["vmin"] = 0.0
            self.cmap.kwargs["vmax"] = source_vmax

        subtracted_img = self.fit.subtracted_images_of_planes_list[final_plane_index]
        self._fit_imaging_meta_plotter._plot_array(subtracted_img, "subtracted_image", "Lens Light Subtracted", ax=axes[5])

        source_model_img = self.fit.model_images_of_planes_list[final_plane_index]
        self._fit_imaging_meta_plotter._plot_array(source_model_img, "source_model_image", "Source Model Image", ax=axes[6])

        if source_vmax is not None:
            self.cmap.kwargs.pop("vmin", None)
            self.cmap.kwargs.pop("vmax", None)

        self.figures_2d_of_planes(
            plane_index=final_plane_index, plane_image=True, use_source_vmax=True, ax=axes[7]
        )

        self.use_log10 = False

        cmap_orig = self.cmap
        if self.residuals_symmetric_cmap:
            self.cmap = self.cmap.symmetric_cmap_from()
        self._fit_imaging_meta_plotter._plot_array(
            self.fit.normalized_residual_map, "normalized_residual_map", "Normalized Residual Map", ax=axes[8]
        )

        self.cmap.kwargs["vmin"] = -1.0
        self.cmap.kwargs["vmax"] = 1.0
        self._fit_imaging_meta_plotter._plot_array(
            self.fit.normalized_residual_map, "normalized_residual_map", r"Normalized Residual Map $1\sigma$", ax=axes[9]
        )
        self.cmap.kwargs.pop("vmin")
        self.cmap.kwargs.pop("vmax")
        self.cmap = cmap_orig

        self.use_log10 = True

        self._fit_imaging_meta_plotter._plot_array(self.fit.chi_squared_map, "chi_squared_map", "Chi-Squared Map", ax=axes[10])

        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True,
            ax=axes[11],
        )

        plt.tight_layout()
        _save_subplot(fig, self.output, f"subplot_fit_log10{plane_index_tag}")

        self.use_log10 = use_log10_orig

    def subplot_of_planes(self, plane_index: Optional[int] = None):
        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:
            fig, axes = plt.subplots(1, 4, figsize=(28, 7))
            axes = axes.flatten()

            self._fit_imaging_meta_plotter._plot_array(self.fit.data, "data", "Data", ax=axes[0])
            self.figures_2d_of_planes(subtracted_image=True, plane_index=plane_index, ax=axes[1])
            self.figures_2d_of_planes(model_image=True, plane_index=plane_index, ax=axes[2])
            self.figures_2d_of_planes(plane_image=True, plane_index=plane_index, ax=axes[3])

            plt.tight_layout()
            _save_subplot(fig, self.output, f"subplot_of_plane_{plane_index}")

    def subplot_tracer(self):
        use_log10_orig = self.use_log10

        final_plane_index = len(self.fit.tracer.planes) - 1

        fig, axes = plt.subplots(3, 3, figsize=(21, 21))
        axes = axes.flatten()

        self._fit_imaging_meta_plotter._plot_array(self.fit.model_data, "model_image", "Model Image", ax=axes[0])

        self.figures_2d_of_planes(
            plane_index=final_plane_index, model_image=True, use_source_vmax=True, ax=axes[1]
        )

        self.figures_2d_of_planes(
            plane_index=final_plane_index,
            plane_image=True,
            zoom_to_brightest=False,
            use_source_vmax=True,
            ax=axes[2],
        )

        tracer_plotter = self.tracer_plotter_of_plane(plane_index=0)
        tracer_plotter._subplot_lens_and_mass(axes=axes, start_index=3)

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_tracer")

        self.use_log10 = use_log10_orig

    def subplot_mappings_of_plane(
        self, plane_index: Optional[int] = None, auto_filename: str = "subplot_mappings"
    ):
        try:
            plane_indexes = self.plane_indexes_from(plane_index=plane_index)

            for plane_index in plane_indexes:
                pixelization_index = 0

                inversion_plotter = self.inversion_plotter_of_plane(plane_index=0)

                fig, axes = plt.subplots(1, 4, figsize=(28, 7))
                axes = axes.flatten()

                inversion_plotter.figures_2d_of_pixelization(
                    pixelization_index=pixelization_index, data_subtracted=True
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
                    plane_index=plane_index, plane_image=True, use_source_vmax=True
                )

                self.figures_2d_of_planes(
                    plane_index=plane_index,
                    plane_image=True,
                    zoom_to_brightest=False,
                    use_source_vmax=True,
                )

                plt.tight_layout()
                _save_subplot(fig, self.output, f"{auto_filename}_{pixelization_index}")
                plt.close(fig)

        except (IndexError, AttributeError, ValueError):
            pass
