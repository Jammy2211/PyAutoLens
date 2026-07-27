"""
Flux-ratio fit component for point-source lensing.

``FitFluxes`` computes the likelihood of the observed image-plane fluxes given the
predicted magnification ratios from the tracer's mass model.

The predicted fluxes are proportional to the absolute magnification at each solved image
position, normalised so that the brightest image has flux 1.0 (flux ratios).  A chi-squared
is computed against the observed flux values and noise map, contributing to the total
``FitPointDataset`` log likelihood.
"""
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


class FitFluxesSolved(AbstractFitPoint):
    """
    Fits the fluxes of a point source dataset with the source-plane flux solved for analytically (in flux space,
    magnification-first), following Lombardi 2024 (arXiv:2406.15280) §6.1, rather than read from a free `flux`
    model parameter.

    With image-plane magnifications `µᵢ` (`magnifications_at_positions`), observed fluxes `f̂ᵢ` and noise `σᵢ`:

        `F* = (Σᵢ µᵢ f̂ᵢ/σᵢ²) / (Σᵢ µᵢ²/σᵢ²)`  (`solved_flux`)

    with model fluxes `µᵢF*` (`model_data`), a standard chi-squared and noise normalization, and the likelihood
    analytically marginalized over `F*` (flat prior):

        `log_likelihood = -0.5*(χ² + noise_norm) - 0.5*log((Σᵢ µᵢ²/σᵢ²)/(2π))`

    The paper's magnitude-space form is not used here: the flux noise maps in this fit are flux-space Gaussians,
    and converting to magnitude space would change the error model, not just its parametrization.

    Works with any profile that has **no** `flux` attribute (`ag.ps.Point` or `ag.ps.PointSolved`); a profile
    with a `flux` attribute (`ag.ps.PointFlux`) raises, since its flux prior would otherwise be sampled by the
    non-linear search but silently ignored by the analytic solve. Use `FitFluxes` for a free-flux fit.
    """

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
        Parameters
        ----------
        name
            The name of the point source dataset which is paired to a `Point` profile.
        data
            The observed fluxes of the point source.
        noise_map
            The noise-map of the fluxes which are used to compute the log likelihood.
        positions
            The image-plane positions of the point source where the fluxes and magnifications are calculated.
        tracer
            The tracer of galaxies whose point source profile is used to fit the fluxes.
        profile
            Manually input the profile of the point source, used instead of one extracted from the tracer.
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

        if hasattr(self.profile, "flux"):
            raise exc.PointExtractionException(
                f"For the point-source named {name} the extracted point source was the class "
                f"{self.profile.__class__.__name__}, which has a `flux` attribute. `FitFluxesSolved` solves "
                f"for the source flux analytically (F*), so a free `flux` prior would be sampled by the "
                f"non-linear search but silently ignored. Use `FitFluxes` with `ag.ps.PointFlux` for a "
                f"free-flux fit, or use a profile with no `flux` attribute (e.g. `ag.ps.Point` / "
                f"`ag.ps.PointSolved`) with `FitFluxesSolved`."
            )

    @property
    def flux_precision_sum(self) -> float:
        """
        `Σᵢ µᵢ²/σᵢ²` — the precision of the solved flux `F*`, and the marginalization normalization.
        """
        mu = self.magnifications_at_positions.array
        sigma_squared = self.noise_map.array**2.0
        return self._xp.sum(mu**2.0 / sigma_squared)

    @property
    def solved_flux(self) -> float:
        """
        `F* = (Σᵢ µᵢ f̂ᵢ/σᵢ²) / (Σᵢ µᵢ²/σᵢ²)`.
        """
        mu = self.magnifications_at_positions.array
        f_hat = self.data.array
        sigma_squared = self.noise_map.array**2.0
        numerator = self._xp.sum(mu * f_hat / sigma_squared)
        return numerator / self.flux_precision_sum

    @property
    def model_data(self) -> aa.ArrayIrregular:
        """
        The model fluxes `µᵢF*`.
        """
        return aa.ArrayIrregular(
            values=self.magnifications_at_positions.array * self.solved_flux
        )

    @property
    def model_fluxes(self) -> aa.ArrayIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the difference between the observed and model fluxes of the point source.
        """
        residual_map = super().residual_map

        return aa.ArrayIrregular(values=residual_map)

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared of the fit of the point source fluxes.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map.array,
        )

    @property
    def marginalization_term(self) -> float:
        """
        The analytic-marginalization contribution to the log likelihood from integrating out the (flat-prior)
        source flux: `-0.5 * log((Σᵢ µᵢ²/σᵢ²)/(2π))`.
        """
        return -0.5 * self._xp.log(self.flux_precision_sum / (2.0 * np.pi))

    @property
    def log_likelihood(self) -> float:
        """
        `log_likelihood = -0.5*(χ² + noise_norm) - 0.5*log((Σᵢ µᵢ²/σᵢ²)/(2π))`.
        """
        return (
            -0.5 * (self.chi_squared + self.noise_normalization)
            + self.marginalization_term
        )
