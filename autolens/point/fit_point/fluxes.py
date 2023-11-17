from functools import partial
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.ray_tracing import Tracer

from autolens import exc


class FitFluxes(aa.FitDataset):
    def __init__(
        self,
        name: str,
        fluxes: aa.ArrayIrregular,
        noise_map: aa.ArrayIrregular,
        positions: aa.Grid2DIrregular,
        tracer: Tracer,
        point_profile: Optional[ag.ps.Point] = None,
    ):
        super().__init__(dataset=fluxes)

        self.tracer = tracer

        self._noise_map = noise_map

        self.name = name
        self.positions = positions

        self.point_profile = (
            tracer.extract_profile(profile_name=name)
            if point_profile is None
            else point_profile
        )

        if self.point_profile is None:
            raise exc.PointExtractionException(
                f"For the point-source named {name} there was no matching point source profile "
                f"in the tracer (make sure your tracer's point source name is the same the dataset name."
            )

        elif not hasattr(self.point_profile, "flux"):
            raise exc.PointExtractionException(
                f"For the point-source named {name} the extracted point source was the "
                f"class {self.point_profile.__class__.__name__} and therefore does "
                f"not contain a flux component."
            )

    @property
    def mask(self):
        return None

    @property
    def noise_map(self):
        return self._noise_map

    @property
    def fluxes(self) -> aa.ArrayIrregular:
        return self.dataset

    @property
    def deflections_func(self):
        """
        Returns the defleciton function, which given the image-plane positions computes their deflection angles.

        For multi-plane ray-tracing with more than 2 planes, the deflection function determines the index of the
        plane with the last mass profile such that the deflection function does not perform unnecessary computations
        beyond this plane.
        """

        if len(self.tracer.planes) > 2:
            upper_plane_index = self.tracer.extract_plane_index_of_profile(
                profile_name=self.name
            )

            return partial(
                self.tracer.deflections_between_planes_from,
                plane_i=0,
                plane_j=upper_plane_index,
            )

        return self.tracer.deflections_yx_2d_from

    @property
    def magnifications(self):
        """
        The magnification of every position in the image-plane, which is computed from the tracer's deflection
        angle map via the Hessian.
        """
        return abs(
            self.tracer.magnification_2d_via_hessian_from(
                grid=self.positions, deflections_func=self.deflections_func
            )
        )

    @property
    def model_data(self):
        """
        The model-fluxes of the tracer at each of the image-plane positions.

        Only point sources which are a `PointFlux` type, and therefore which include a model parameter for its flux,
        are used.
        """
        return aa.ArrayIrregular(
            values=[
                magnification * self.point_profile.flux
                for magnification in self.magnifications
            ]
        )

    @property
    def model_fluxes(self) -> aa.ArrayIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the residual map, over riding the parent method so that the result is converted to a
        `ArrayIrregular` object.
        """
        residual_map = super().residual_map

        return aa.ArrayIrregular(values=residual_map)

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map,
        )
