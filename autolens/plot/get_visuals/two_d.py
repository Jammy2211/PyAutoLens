from typing import List, Union

import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

from autogalaxy.plot.get_visuals import one_d as gv1d
from autogalaxy.plot.get_visuals import two_d as gv2d

from autolens.imaging.fit_imaging import FitImaging
from autolens.lens.ray_tracing import Tracer


class GetVisuals2D(gv2d.GetVisuals2D):
    def __init__(self, include: aplt.Include2D, visuals: aplt.Visuals2D):
        """
        Class which gets 2D attributes and adds them to a `Visuals2D` objects, such that they are plotted on 2D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include2D` object. If this entry is `False`, the `GetVisuals2D.get` method returns a None and the
        attribute is omitted from the plot.

        The `GetVisuals2D` class adds new visuals to a pre-existing `Visuals2D` object that is passed to
        its `__init__` method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 2D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals2D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals2D` class.
        """
        super().__init__(include=include, visuals=visuals)

    def via_tracer_from(
        self, tracer: Tracer, grid: aa.type.Grid2DLike, plane_index: int
    ) -> aplt.Visuals2D:
        """
        From a `Tracer` get the attributes that can be plotted and returns them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted.

        From a tracer the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the light object's quantities in 2D.
        - border: the border of the mask of the grid used to plot the light object's quantities in 2D.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - tangential_critical curves: the tangential critical curves of all of the tracer's mass profiles combined.
        - tangential_caustics: the tangential caustics of all of the tracer's mass profiles combined.
        - radial_critical curves: the radial critical curves of all of the tracer's mass profiles combined.
        - radial_caustics: the radial caustics of all of the tracer's mass profiles combined.

        When plotting a `Tracer` it is common for plots to only display quantities corresponding to one plane at a time
        (e.g. the convergence in the image plane, the source in the source plane). Therefore, quantities are only
        extracted from one plane, specified by the  input `plane_index`.

        Parameters
        ----------
        tracer
            The `Tracer` object which has attributes extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the tracer's quantities in 2D.
        plane_index
            The index of the plane in the tracer which is used to extract quantities, as only one plane is plotted
            at a time.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """
        origin = self.get("origin", value=aa.Grid2DIrregular(values=[grid.origin]))

        border = self.get("border", value=grid.mask.derive_grid.border_sub_1.binned)

        if border is not None and len(border) > 0 and plane_index > 0:
            border = tracer.traced_grid_2d_list_from(grid=border)[plane_index]

        light_profile_centres = self.get(
            "light_profile_centres",
            tracer.planes[plane_index].extract_attribute(
                cls=ag.LightProfile, attr_name="centre"
            ),
        )

        mass_profile_centres = self.get(
            "mass_profile_centres",
            tracer.planes[plane_index].extract_attribute(
                cls=ag.mp.MassProfile, attr_name="centre"
            ),
        )

        tangential_critical_curves = None
        radial_critical_curves = None
        tangential_caustics = None
        radial_caustics = None

        if plane_index == 0:
            tangential_critical_curves = self.get(
                "tangential_critical_curves",
                tracer.tangential_critical_curve_list_from(grid=grid),
                "tangential_critical_curves",
            )

            radial_critical_curves = None

            radial_critical_curve_area_list = (
                tracer.radial_critical_curve_area_list_from(grid=grid)
            )

            if any(
                [area > grid.pixel_scale for area in radial_critical_curve_area_list]
            ):
                radial_critical_curves = self.get(
                    "radial_critical_curves",
                    tracer.radial_critical_curve_list_from(grid=grid),
                    "radial_critical_curves",
                )

        if plane_index > 0:
            tangential_caustics = self.get(
                "tangential_caustics",
                tracer.tangential_caustic_list_from(grid=grid),
                "tangential_caustics",
            )

            radial_caustics = self.get(
                "radial_caustics",
                tracer.radial_caustic_list_from(grid=grid),
                "radial_caustics",
            )

        return self.visuals + self.visuals.__class__(
            origin=origin,
            border=border,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            tangential_critical_curves=tangential_critical_curves,
            tangential_caustics=tangential_caustics,
            radial_critical_curves=radial_critical_curves,
            radial_caustics=radial_caustics,
        )

    def via_fit_imaging_from(self, fit: FitImaging) -> aplt.Visuals2D:
        """
        From a `FitImaging` get its attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `FitImaging` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 2D coordinate system.
        - mask: the 2D mask.
        - border: the border of the 2D mask, which are all of the mask's exterior edge pixels.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - critical curves: the critical curves of all mass profile combined.

        Parameters
        ----------
        fit
            The fit imaging object whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        visuals_2d_via_mask = self.via_mask_from(mask=fit.mask)

        visuals_2d_via_tracer = self.via_tracer_from(
            tracer=fit.tracer, grid=fit.grid, plane_index=0
        )

        return visuals_2d_via_mask + visuals_2d_via_tracer
