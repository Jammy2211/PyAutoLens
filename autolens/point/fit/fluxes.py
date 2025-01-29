from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.point.fit.abstract import AbstractFitPoint
from autolens.lens.tracer import Tracer

from autolens import exc


class FitFluxes(AbstractFitPoint):
    def __init__(
        self,
        name: str,
        data: aa.ArrayIrregular,
        noise_map: aa.ArrayIrregular,
        positions: aa.Grid2DIrregular,
        tracer: Tracer,
        profile: Optional[ag.ps.Point] = None,
    ):

        self.positions = positions

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=None,
            profile=profile,
        )

        if not hasattr(self.profile, "flux"):
            raise exc.PointExtractionException(
                f"For the point-source named {name} the extracted point source was the "
                f"class {self.profile.__class__.__name__} and therefore does "
                f"not contain a flux component."
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
                magnification * self.profile.flux
                for magnification in self.magnifications_at_positions
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
