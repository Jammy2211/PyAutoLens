"""
Time-delay fit component for point-source lensing.

``FitTimeDelays`` computes the likelihood of the observed time delays between multiple
images of a point source given the Fermat potential predicted by the tracer's mass and
light models.

The time delay between images i and j is:

  Δt_{ij} = (D_Δt / c) × [φ(θ_i) − φ(θ_j)]

where D_Δt is the time-delay distance computed from the cosmology and φ is the Fermat
potential.  A chi-squared is computed against the observed delays and their noise map,
contributing to the total ``FitPointDataset`` log likelihood.
"""
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.point.fit.abstract import AbstractFitPoint
from autolens.lens.tracer import Tracer


class FitTimeDelays(AbstractFitPoint):
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
        Fits the time delays of a point source dataset using a `Tracer` object,
        where every model time delay of the point-source is compared with its observed time delay.

        The fit performs the following steps:

        1) Compute the model time delays at the input image-plane `positions` using the tracer.

        2) Compute the relative time delays of the dataset time delays and the time delays of the point source at
           these positions, which are the time delays  relative to the shortest time delay

        2) Subtract the observed relative time delays from the model relative time delays to compute the residuals.

        3) Compute the chi-squared of each time delay residual.

        4) Sum the chi-squared values to compute the overall log likelihood of the fit.

        Time delay fitting uses name pairing similar to flux fitting to ensure
        the correct point source profile is used.

        Parameters
        ----------
        name
            The name of the point source dataset which is paired to a `Point` profile.
        data
            The observed time delays in days of the point source.
        noise_map
            The noise-map of the time delays in days used to compute the log likelihood.
        tracer
            The tracer of galaxies whose point source profile is used to fit the time delays.
        positions
            The image-plane positions of the point source where the time delays are calculated.
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

    @property
    def model_data(self) -> aa.ArrayIrregular:
        """
        The model time delays of the tracer at each of the input image-plane positions.

        These values are not subtracted by the shorter time delay of the point source, which would make the shorter
        delay have a value of zero. However, this subtraction is performed in the `residual_map` property, in order
        to ensure the residuals are computed relative to the shorter time delay.
        """
        return self.tracer.time_delays_from(grid=self.positions, xp=self._xp)

    @property
    def model_time_delays(self) -> aa.ArrayIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the difference between the observed and model time delays of the point source,
        which is the residual time delay of the fit.

        The residuals are computed relative to the shortest time delay of the point source, which is subtracted
        from the dataset time delays and model time delays before the subtraction.
        """

        data = self.data - self._xp.min(self.data.array)
        model_data = self.model_data - self._xp.min(self.model_data.array)

        residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
        return aa.ArrayIrregular(values=residual_map)

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared of the fit of the point source time delays,
        which is the residual values divided by the RMS noise-map squared.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map.array,
        )


class FitTimeDelaysSolved(FitTimeDelays):
    """
    Fits the time delays of a point source dataset with the reference time solved for analytically, following
    Lombardi 2024 (arXiv:2406.15280) §6.1, rather than by subtracting the shortest delay from both the data and
    the model (`FitTimeDelays.residual_map`, left unchanged for continuity).

    With model per-image delays `Tᵢ` (`tracer.time_delays_from`, `model_data`, inherited unchanged) and per-image
    precision `τᵢ = σᵢ⁻²` (`tau`):

        `T* = Σᵢ τᵢ(t̂ᵢ − Tᵢ) / Σᵢ τᵢ`  (`solved_reference_time`)

    with residuals `t̂ᵢ − (Tᵢ + T*)` (`residual_map`), a standard chi-squared and noise normalization, and the
    likelihood analytically marginalized over `T*` (flat prior):

        `log_likelihood = -0.5*(χ² + noise_norm) - 0.5*log((Σᵢ τᵢ)/(2π))`

    Does not depend on which profile is paired to the dataset (time delays are computed from the tracer alone),
    so works with `ag.ps.Point`, `ag.ps.PointFlux` or `ag.ps.PointSolved` interchangeably.
    """

    @property
    def tau(self) -> np.ndarray:
        """
        `τᵢ = σᵢ⁻²` — the per-image time-delay precision.
        """
        return self.noise_map.array**-2.0

    @property
    def tau_sum(self) -> float:
        """
        `Σᵢ τᵢ` — the precision of the solved reference time `T*`, and the marginalization normalization.
        """
        return self._xp.sum(self.tau)

    @property
    def solved_reference_time(self) -> float:
        """
        `T* = Σᵢ τᵢ(t̂ᵢ − Tᵢ) / Σᵢ τᵢ`.
        """
        t_hat = self.data.array
        model_delays = self.model_data.array
        return self._xp.sum(self.tau * (t_hat - model_delays)) / self.tau_sum

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the difference between the observed time delays and the model time delays offset by the solved
        reference time: `t̂ᵢ − (Tᵢ + T*)`.
        """
        residual_map = self.data.array - (
            self.model_data.array + self.solved_reference_time
        )
        return aa.ArrayIrregular(values=residual_map)

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared of the fit of the point source time delays.
        """
        return ag.util.fit.chi_squared_from(
            chi_squared_map=self.chi_squared_map.array,
        )

    @property
    def marginalization_term(self) -> float:
        """
        The analytic-marginalization contribution to the log likelihood from integrating out the (flat-prior)
        reference time: `-0.5 * log((Σᵢ τᵢ)/(2π))`.
        """
        return -0.5 * self._xp.log(self.tau_sum / (2.0 * np.pi))

    @property
    def log_likelihood(self) -> float:
        """
        `log_likelihood = -0.5*(χ² + noise_norm) - 0.5*log((Σᵢ τᵢ)/(2π))`.
        """
        return (
            -0.5 * (self.chi_squared + self.noise_normalization)
            + self.marginalization_term
        )
