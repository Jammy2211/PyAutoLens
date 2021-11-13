from typing import List, Union

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autogalaxy.plot.mat_wrap import get_visuals as gv

from autolens.imaging.fit_imaging import FitImaging
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.ray_tracing import Tracer


class GetVisuals1D(gv.GetVisuals1D):
    def __init__(self, include: aplt.Include1D, visuals: aplt.Visuals1D):

        super().__init__(include=include, visuals=visuals)


class GetVisuals2D(gv.GetVisuals2D):
    def __init__(self, include: aplt.Include2D, visuals: aplt.Visuals2D):

        super().__init__(include=include, visuals=visuals)

    def via_tracer_from(
        self, tracer: Tracer, grid: aa.type.Grid2DLike, plane_index: int
    ) -> aplt.Visuals2D:

        origin = self.get("origin", value=aa.Grid2DIrregular(grid=[grid.origin]))

        border = self.get("border", value=grid.mask.border_grid_sub_1.binned)

        if border is not None:
            if plane_index > 0:
                border = tracer.traced_grids_of_planes_from(grid=border)[plane_index]

        light_profile_centres = self.get(
            "light_profile_centres",
            tracer.planes[plane_index].extract_attribute(
                cls=ag.lp.LightProfile, attr_name="centre"
            ),
        )

        mass_profile_centres = self.get(
            "mass_profile_centres",
            tracer.planes[plane_index].extract_attribute(
                cls=ag.mp.MassProfile, attr_name="centre"
            ),
        )

        if plane_index == 0:
            critical_curves = self.get(
                "critical_curves",
                tracer.critical_curves_from(grid=grid),
                "critical_curves",
            )
        else:
            critical_curves = None

        if plane_index == 1:
            caustics = self.get("caustics", tracer.caustics_from(grid=grid), "caustics")
        else:
            caustics = None

        return self.visuals + self.visuals.__class__(
            origin=origin,
            border=border,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            critical_curves=critical_curves,
            caustics=caustics,
        )

    def via_fit_imaging_from(self, fit: FitImaging) -> aplt.Visuals2D:

        visuals_2d_via_mask = self.via_mask_from(mask=fit.mask)

        visuals_2d_via_tracer = self.via_tracer_from(
            tracer=fit.tracer, grid=fit.grid, plane_index=0
        )

        return visuals_2d_via_mask + visuals_2d_via_tracer
