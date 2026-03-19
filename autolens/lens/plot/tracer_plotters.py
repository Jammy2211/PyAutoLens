import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autoconf import cached_property
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autogalaxy.plot.mass_plotter import MassPlotter
from autogalaxy.plot.abstract_plotters import _save_subplot

from autolens.plot.abstract_plotters import Plotter, _to_lines, _to_positions
from autolens.lens.tracer import Tracer

from autolens import exc

from autolens.lens import tracer_util


class TracerPlotter(Plotter):
    def __init__(
        self,
        tracer: Tracer,
        grid: aa.type.Grid2DLike,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
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

        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

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
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
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
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
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
        ax=None,
    ):
        if image:
            self._plot_array(
                array=self.tracer.image_2d_from(grid=self.grid),
                auto_filename="image_2d",
                title="Image",
                lines=self._lines_for_image_plane(),
                positions=_to_positions(self.positions),
                ax=ax,
            )

        if source_plane:
            self.figures_2d_of_planes(
                plane_image=True, plane_index=len(self.tracer.planes) - 1, ax=ax,
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
        ax=None,
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
                    ax=ax,
                )

            if plane_grid:
                galaxies_plotter.figures_2d(
                    plane_grid=True,
                    title_suffix=f" Of Plane {plane_index}",
                    filename_suffix=f"_of_plane_{plane_index}",
                    source_plane_title=source_plane_title,
                    ax=ax,
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
        items = [
            (image, "image"),
            (source_plane, "source_plane"),
            (convergence, "convergence"),
            (potential, "potential"),
            (deflections_y, "deflections_y"),
            (deflections_x, "deflections_x"),
            (magnification, "magnification"),
        ]
        n = sum(1 for flag, _ in items if flag)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

        idx = 0
        if image:
            self.figures_2d(image=True, ax=axes_flat[idx])
            idx += 1
        if source_plane:
            self.figures_2d(source_plane=True, ax=axes_flat[idx])
            idx += 1
        if convergence:
            self.figures_2d(convergence=True, ax=axes_flat[idx])
            idx += 1
        if potential:
            self.figures_2d(potential=True, ax=axes_flat[idx])
            idx += 1
        if deflections_y:
            self.figures_2d(deflections_y=True, ax=axes_flat[idx])
            idx += 1
        if deflections_x:
            self.figures_2d(deflections_x=True, ax=axes_flat[idx])
            idx += 1
        if magnification:
            self.figures_2d(magnification=True, ax=axes_flat[idx])
            idx += 1

        plt.tight_layout()
        _save_subplot(fig, self.output, auto_filename)

    def subplot_tracer(self):
        final_plane_index = len(self.tracer.planes) - 1

        fig, axes = plt.subplots(3, 3, figsize=(21, 21))
        axes = axes.flatten()

        self._plot_array(
            array=self.tracer.image_2d_from(grid=self.grid),
            auto_filename="image_2d",
            title="Image",
            lines=self._lines_for_image_plane(),
            ax=axes[0],
        )

        galaxies_plotter_no_caustics = self.galaxies_plotter_from(
            plane_index=final_plane_index, include_caustics=False
        )
        galaxies_plotter_no_caustics.figures_2d(
            image=True, title_suffix="", ax=axes[1]
        )

        galaxies_plotter_source = self.galaxies_plotter_from(
            plane_index=final_plane_index, include_caustics=True
        )
        galaxies_plotter_source.figures_2d(
            plane_image=True,
            title_suffix=f" Of Plane {final_plane_index}",
            filename_suffix=f"_of_plane_{final_plane_index}",
            source_plane_title=True,
            ax=axes[2],
        )

        self._subplot_lens_and_mass(axes=axes, start_index=3)

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_tracer")

    def _subplot_lens_and_mass(self, axes, start_index: int = 0):
        use_log10_orig = self.use_log10
        self.use_log10 = True

        galaxies_plotter = self.galaxies_plotter_from(plane_index=0)
        if start_index < len(axes):
            galaxies_plotter.figures_2d(
                image=True,
                title_suffix=" Of Plane 0",
                ax=axes[start_index],
            )

        self.use_log10 = use_log10_orig

    def subplot_lensed_images(self):
        number_subplots = self.tracer.total_planes

        fig, axes = plt.subplots(1, number_subplots, figsize=(7 * number_subplots, 7))
        axes_flat = [axes] if number_subplots == 1 else list(np.array(axes).flatten())

        for plane_index in range(0, self.tracer.total_planes):
            galaxies_plotter = self.galaxies_plotter_from(plane_index=plane_index)
            galaxies_plotter.figures_2d(
                image=True,
                title_suffix=f" Of Plane {plane_index}",
                ax=axes_flat[plane_index],
            )

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_lensed_images")

    def subplot_galaxies_images(self):
        # Layout: plane 0 image + for each plane>0: lensed image + source plane image
        # But the skip in old code = 2 * total_planes - 1 total slots
        n = 2 * self.tracer.total_planes - 1

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

        idx = 0
        galaxies_plotter = self.galaxies_plotter_from(plane_index=0)
        if idx < n:
            galaxies_plotter.figures_2d(
                image=True, title_suffix=" Of Plane 0", ax=axes_flat[idx]
            )
            idx += 1

        for plane_index in range(1, self.tracer.total_planes):
            galaxies_plotter = self.galaxies_plotter_from(plane_index=plane_index)
            if idx < n:
                galaxies_plotter.figures_2d(
                    image=True,
                    title_suffix=f" Of Plane {plane_index}",
                    ax=axes_flat[idx],
                )
                idx += 1
            if idx < n:
                galaxies_plotter.figures_2d(
                    plane_image=True,
                    title_suffix=f" Of Plane {plane_index}",
                    ax=axes_flat[idx],
                )
                idx += 1

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_galaxies_images")
