from typing import Optional, List

import numpy as np

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autoconf import cached_property
from autogalaxy.plot.mass_plotter import MassPlotter

from autolens.plot.abstract_plotters import Plotter, _to_lines, _to_positions
from autolens.lens.tracer import Tracer

from autolens import exc

from autolens.lens import tracer_util


class TracerPlotter(Plotter):
    def __init__(
        self,
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
        mat_plot_1d: aplt.MatPlot1D = None,
        mat_plot_2d: aplt.MatPlot2D = None,
        positions=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
        tangential_caustics=None,
        radial_caustics=None,
    ):
        from autogalaxy.profiles.light.linear import LightProfileLinear

        if tracer.has(cls=LightProfileLinear):
            raise exc.raise_linear_light_profile_in_plot(
                plotter_type=self.__class__.__name__,
            )

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        self.tracer = tracer
        self.grid = grid
        self.positions = positions

        self._tc = tangential_critical_curves
        self._rc = radial_critical_curves
        self._tc_caustic = tangential_caustics
        self._rc_caustic = radial_caustics

        self._mass_plotter = MassPlotter(
            mass_obj=self.tracer,
            grid=self.grid,
            mat_plot_2d=self.mat_plot_2d,
            tangential_critical_curves=tangential_critical_curves,
            radial_critical_curves=radial_critical_curves,
        )

    # ------------------------------------------------------------------
    # Cached critical-curve / caustic helpers (computed via LensCalc)
    # ------------------------------------------------------------------

    @cached_property
    def _critical_curves_pair(self):
        tan_cc, rad_cc = tracer_util.critical_curves_from(
            tracer=self.tracer, grid=self.grid
        )
        return list(tan_cc), list(rad_cc)

    @cached_property
    def _caustics_pair(self):
        tan_ca, rad_ca = tracer_util.caustics_from(
            tracer=self.tracer, grid=self.grid
        )
        return list(tan_ca), list(rad_ca)

    @property
    def tangential_critical_curves(self):
        if self._tc is not None:
            return self._tc
        return self._critical_curves_pair[0]

    @property
    def radial_critical_curves(self):
        if self._rc is not None:
            return self._rc
        return self._critical_curves_pair[1]

    @property
    def tangential_caustics(self):
        if self._tc_caustic is not None:
            return self._tc_caustic
        return self._caustics_pair[0]

    @property
    def radial_caustics(self):
        if self._rc_caustic is not None:
            return self._rc_caustic
        return self._caustics_pair[1]

    def _lines_for_image_plane(self) -> Optional[List[np.ndarray]]:
        return _to_lines(self.tangential_critical_curves, self.radial_critical_curves)

    def _lines_for_source_plane(self) -> Optional[List[np.ndarray]]:
        return _to_lines(self.tangential_caustics, self.radial_caustics)

    def galaxies_plotter_from(
        self, plane_index: int, include_caustics: bool = True
    ) -> aplt.GalaxiesPlotter:
        plane_grid = self.tracer.traced_grid_2d_list_from(grid=self.grid)[plane_index]

        if plane_index == 0:
            tc = self.tangential_critical_curves
            rc = self.radial_critical_curves
            tc_ca = None
            rc_ca = None
        else:
            tc = None
            rc = None
            if include_caustics:
                tc_ca = self.tangential_caustics
                rc_ca = self.radial_caustics
            else:
                tc_ca = None
                rc_ca = None

        return aplt.GalaxiesPlotter(
            galaxies=ag.Galaxies(galaxies=self.tracer.planes[plane_index]),
            grid=plane_grid,
            mat_plot_2d=self.mat_plot_2d,
            tangential_critical_curves=tc if tc is not None else tc_ca,
            radial_critical_curves=rc if rc is not None else rc_ca,
        )

    def figures_2d(
        self,
        image: bool = False,
        source_plane: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
    ):
        if image:
            self._plot_array(
                array=self.tracer.image_2d_from(grid=self.grid),
                auto_labels=aplt.AutoLabels(title="Image", filename="image_2d"),
                lines=self._lines_for_image_plane(),
                positions=_to_positions(self.positions),
            )

        if source_plane:
            self.figures_2d_of_planes(
                plane_image=True, plane_index=len(self.tracer.planes) - 1
            )

        self._mass_plotter.figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

    def plane_indexes_from(self, plane_index: Optional[int]) -> List[int]:
        if plane_index is None:
            return list(range(len(self.tracer.planes)))
        return [plane_index]

    def figures_2d_of_planes(
        self,
        plane_image: bool = False,
        plane_grid: bool = False,
        plane_index: Optional[int] = None,
        zoom_to_brightest: bool = True,
        include_caustics: bool = True,
    ):
        plane_indexes = self.plane_indexes_from(plane_index=plane_index)

        for plane_index in plane_indexes:
            galaxies_plotter = self.galaxies_plotter_from(
                plane_index=plane_index, include_caustics=include_caustics
            )

            source_plane_title = plane_index == 1

            if plane_image:
                galaxies_plotter.figures_2d(
                    plane_image=True,
                    zoom_to_brightest=zoom_to_brightest,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
                    source_plane_title=source_plane_title,
                )

            if plane_grid:
                galaxies_plotter.figures_2d(
                    plane_grid=True,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
                    source_plane_title=source_plane_title,
                )

    def subplot(
        self,
        image: bool = False,
        source_plane: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        auto_filename: str = "subplot_tracer",
    ):
        self._subplot_custom_plot(
            image=image,
            source_plane=source_plane,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_tracer(self):
        final_plane_index = len(self.tracer.planes) - 1

        use_log10_original = self.mat_plot_2d.use_log10

        self.open_subplot_figure(number_subplots=9)

        self.figures_2d(image=True)

        self.set_title(label="Lensed Source Image")

        # Show lensed source image without caustics
        galaxies_plotter = self.galaxies_plotter_from(
            plane_index=final_plane_index, include_caustics=False
        )
        galaxies_plotter.figures_2d(image=True)

        self.set_title(label="Source Plane Image")
        self.figures_2d(source_plane=True)
        self.set_title(label=None)

        self._subplot_lens_and_mass()

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_tracer")
        self.close_subplot_figure()

        self.mat_plot_2d.use_log10 = use_log10_original

    def _subplot_lens_and_mass(self):

        self.mat_plot_2d.use_log10 = True

        self.set_title(label="Lens Galaxy Image")

        self.figures_2d_of_planes(
            plane_image=True,
            plane_index=0,
            zoom_to_brightest=False,
        )

        self.mat_plot_2d.subplot_index = 5

        self.set_title(label=None)
        self.figures_2d(convergence=True)

        self.figures_2d(potential=True)

        self.mat_plot_2d.use_log10 = False

        self.figures_2d(magnification=True)
        self.figures_2d(deflections_y=True)
        self.figures_2d(deflections_x=True)

    def subplot_lensed_images(self):
        number_subplots = self.tracer.total_planes

        self.open_subplot_figure(number_subplots=number_subplots)

        for plane_index in range(0, self.tracer.total_planes):
            galaxies_plotter = self.galaxies_plotter_from(plane_index=plane_index)
            galaxies_plotter.figures_2d(
                image=True, title_suffix=f" Of Plane {plane_index}"
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_lensed_images"
        )
        self.close_subplot_figure()

    def subplot_galaxies_images(self):
        number_subplots = 2 * self.tracer.total_planes - 1

        self.open_subplot_figure(number_subplots=number_subplots)

        galaxies_plotter = self.galaxies_plotter_from(plane_index=0)
        galaxies_plotter.figures_2d(image=True, title_suffix=" Of Plane 0")

        self.mat_plot_2d.subplot_index += 1

        for plane_index in range(1, self.tracer.total_planes):
            galaxies_plotter = self.galaxies_plotter_from(plane_index=plane_index)
            galaxies_plotter.figures_2d(
                image=True, title_suffix=f" Of Plane {plane_index}"
            )
            galaxies_plotter.figures_2d(
                plane_image=True, title_suffix=f" Of Plane {plane_index}"
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_galaxies_images"
        )
        self.close_subplot_figure()
