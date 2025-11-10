import numpy as np
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
        xp=np,
    ):
        """
        Fits the fluxes of a a point source dataset using a `Tracer` object, where every model flux of the point-source
        is compared with its observed flux.

        The fit performs the following steps:

        1) Compute the magnification of the input image-plane `positions` via the Hessian of the tracer's deflection angles.

        2) Determine the image-plane model fluxes by multiplying the source-plane flux with these magnifications.

        3) Subtract the observed fluxes from the model fluxes to compute the residual fluxes, called the `residual_map`.

        4) Compute the chi-squared of each flux as the square of the residual divided by the RMS noise-map value.

        5) Sum the chi-squared values to compute the overall log likelihood of the fit.

        Flux based fitting in the source code always inputs the observed positions of the point dataset as the input
        `positions`, but the following changes could be implemented and used in the future:

        - Use the model positions instead of the observed positions to compute the fluxes, which would therefore
          require the centre of the point source in the source-plane to be used and for the `PointSolver` to determine
          the image-plane positions via ray-tracing triangles to and from the source-plane. This would require
          care in pairing model positions to observed positions where fluxes are computed.

        - The "size" of the point-source is not currently supported, however the `ShapeSolver` implemented in the
          source code does allow for magnifications to be computed based on point sources with a shape (e.g. a
          `Circle` where its radius is a free parameter).

        Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
        point source dataset to ensure that point source datasets are fitted to the correct point source.

        This fit object is used in the `FitPointDataset` to perform position based fitting of a `PointDataset`,
        which may also fit other components of the point dataset like fluxes or time delays.

        When performing a `model-fit` via an `AnalysisPoint` object the `figure_of_merit` of this object
        is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        name
            The name of the point source dataset which is paired to a `Point` profile.
        data
            The positions of the point source in the image-plane which are fitted.
        noise_map
            The noise-map of the positions which are used to compute the log likelihood of the positions.
        tracer
            The tracer of galaxies whose point source profile are used to fit the positions.
        positions
            The positions of the point source in the image-plane where the fluxes are calculated. These are currently
            always the observed positions of the point source in the source code, but other positions, like the
            model positions, could be used in the future.
        profile
            Manually input the profile of the point source, which is used instead of the one extracted from the
            tracer via name pairing if that profile is not found.
        """
        self.positions = positions

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=None,
            profile=profile,
            xp=xp,
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
        The model-fluxes of the tracer at each of the input image-plane positions.

        Only point sources which are a `PointFlux` type, and therefore which include a model parameter for its flux,
        are used.
        """
        return aa.ArrayIrregular(
            values=self._xp.array(
                [
                    magnification * self.profile.flux
                    for magnification in self.magnifications_at_positions
                ]
            )
        )

    @property
    def model_fluxes(self) -> aa.ArrayIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the difference between the observed and model fluxes of the point source, which is the residual flux
        of a point source flux fit.
        """
        residual_map = super().residual_map

        return aa.ArrayIrregular(values=residual_map)

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared of the fit of the point source fluxes, which is the residual flux values divided by the
        RMS noise-map values squared.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map.array,
        )
