"""
Source-plane position fitting via traced-position separations.

Instead of comparing predicted and observed positions in the image plane,
``FitPositionsSourcePlane`` traces each *observed* image position back to the source
plane via the tracer's deflection angles and measures how tightly they converge.

If the lens model is correct, all observed images of the same source should trace back
to (approximately) the same source-plane coordinate.  The figure of merit is the squared
separation of the back-traced positions from that source-plane coordinate, normalised by
the position noise map.  That coordinate is not a computed centroid of the back-traced
positions: it is the paired point-source profile's ``centre`` (a free model parameter, for
``FitPositionsSource``) or, for ``FitPositionsSourceSolved``, the analytically-solved centre
``β*`` (see ``autolens.point.fit.solved.SolvedCentre``).

This approach avoids the need for a ``PointSolver`` (no forward-solving is required) and
is well-suited to JAX-accelerated model fits.
"""
import numpy as np
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.point.fit.solved import SolvedCentre, precision_tensor_components_from
from autolens.point.solver import PointSolver


class FitPositionsSource(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: Optional[PointSolver],
        profile: Optional[ag.ps.Point] = None,
        xp=np,
    ):
        """
        Fits the positions of a a point source dataset using a `Tracer` object with a source-plane chi-squared based on
        the separation of image-plane positions ray-traced to the source-plane compared to the centre of the source
        galaxy.

        The fit performs the following steps:

        1) Determine the source-plane centre of the source-galaxy, which is either a free model parameter read
           from the profile's `centre` (`ag.ps.Point` / `ag.ps.PointFlux`) or, for `FitPositionsSourceSolved`,
           solved for analytically given the current tracer (see `autolens.point.fit.solved.SolvedCentre`),
           using name pairing (see below).

        2) Ray-trace the positions in the point source to the source-plane via the `Tracer`, including accounting for
           multi-plane ray-tracing.

        3) Compute the distance of each ray-traced position to the source-plane centre and compute the residuals,

        4) Compute the magnification of each image-plane position via the Hessian of the tracer's deflection angles.

        5) Compute the residuals of each position as the difference between the source-plane centre and each
           ray-traced position.

        6) Compute the chi-squared of each position as the square of the residual multiplied by the magnification and
           divided by the RMS noise-map value.

        7) Sum the chi-squared values to compute the overall log likelihood of the fit.

        Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
        point source dataset to ensure that point source datasets are fitted to the correct point source.

        This fit object is used in the `FitPointDataset` to perform position based fitting of a `PointDataset`,
        which may also fit other components of the point dataset like fluxes or time delays.

        When performing a `model-fit`via an `AnalysisPoint` object the `figure_of_merit` of this object
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
        solver
            Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
            triangles to and from the source-plane. This is not used in this source-plane point source fit.
        profile
            Manually input the profile of the point source, which is used instead of the one extracted from the
            tracer via name pairing if that profile is not found.
        """

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
            xp=xp,
        )

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the source-plane model positions of the point source, which are the positions of the image-plane
        positions ray-traced to the source-plane.

        This calculation accounts for multi-plane ray-tracing, whereby if the tracer has more than 2 planees the
        redshift of the point source galaxy is extracted and the deflections between the image-plane and source-plane
        at its specific redshift are used.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(
                grid=self.data, xp=self._xp
            )
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, xp=self._xp, plane_i=0, plane_j=self.plane_index
            )

        return self.data.grid_2d_via_deflection_grid_from(
            deflection_grid=deflections, xp=self._xp
        )

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the residuals of the point-source source-plane fit, which are the distances of each source-plane
        position from the source-plane centre.
        """
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )

    @property
    def chi_squared_map(self) -> float:
        """
        Returns the chi-squared of the point-source source-plane fit, which is the sum of the squared residuals
        multiplied by the magnifications squared, divided by the noise-map values squared.
        """

        return self.residual_map**2.0 / (
            self.magnifications_at_positions.array**-2.0 * self.noise_map.array**2.0
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the normalization of the noise-map, which is the sum of the noise-map values squared.
        """
        return self._xp.sum(
            self._xp.log(
                2
                * np.pi
                * (
                    self.magnifications_at_positions.array**-2.0
                    * self.noise_map.array**2.0
                )
            )
        )

    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of the point-source source-plane fit, which is the sum of the chi-squared values.
        """
        return -0.5 * (sum(self.chi_squared_map) + self.noise_normalization)


class FitPositionsSourceSolved(SolvedCentre, FitPositionsSource):
    """
    ``FitPositionsSource`` with the source-plane centre solved for analytically, rather than read from a free
    ``centre`` model parameter, following Lombardi 2024 (arXiv:2406.15280) §5.1.

    The lens equation is Taylor-expanded around each observed image position `θ̂ᵢ`, giving a source-plane
    position that is linear in the per-image back-traced position `β̂ᵢ` (`SolvedCentre._beta_hat`, computed the
    same way as `FitPositionsSource.model_data`) and the per-image precision `Wᵢ` (`weighting = "jacobian"` by
    default — the tensor weighting `Wᵢ = Aᵢ⁻ᵀΘᵢAᵢ⁻¹`; set `weighting = "magnification"` for the scalar
    `Wᵢ = (µᵢ²/σᵢ²) I₂` weighting used by `FitPositionsSource`, for Lenstool-style comparisons). This makes the
    source-plane centre solvable in closed form (`SolvedCentre.source_plane_coordinate`, `β*`), and the
    likelihood analytically marginalizes over it (flat prior):

        `log_likelihood = -0.5*(χ² + noise_norm) - 0.5*log(det(Σᵢ Wᵢ) / (2π)²)`

    where `χ² = Σᵢ (β̂ᵢ−β*)ᵀ Wᵢ (β̂ᵢ−β*)` (`chi_squared_map` / `chi_squared`) and
    `noise_norm = Σᵢ log((2π)²/det Wᵢ)` (`noise_normalization`), each a separately-testable property, alongside
    the marginalization term itself (`marginalization_term`).

    Must be paired (by name) with a parameter-free profile such as `ag.ps.PointSolved`: a `centre`-bearing
    profile (`ag.ps.Point` / `ag.ps.PointFlux`) raises (see `SolvedCentre.source_plane_coordinate`), since its
    centre priors would otherwise be sampled but silently ignored.
    """

    _non_solved_alternative_name = "FitPositionsSource"

    @property
    def residual_vectors(self) -> np.ndarray:
        """
        The (n_positions, 2) array of vector residuals `β̂ᵢ − β*`, i.e. the back-traced source-plane positions
        minus the solved source-plane centre.
        """
        beta_hat = self._beta_hat.array
        beta_star_y, beta_star_x = self.source_plane_coordinate
        beta_star = self._xp.array([beta_star_y, beta_star_x])
        return beta_hat - beta_star

    @property
    def chi_squared_map(self) -> aa.ArrayIrregular:
        """
        The per-image quadratic form `(β̂ᵢ−β*)ᵀ Wᵢ (β̂ᵢ−β*)`.
        """
        w11, w12, w21, w22 = precision_tensor_components_from(self, self.weighting)

        delta = self.residual_vectors
        dy = delta[:, 0]
        dx = delta[:, 1]

        terms = dy * (w11 * dy + w12 * dx) + dx * (w21 * dy + w22 * dx)

        return aa.ArrayIrregular(values=terms)

    @property
    def chi_squared(self) -> float:
        """
        `χ² = Σᵢ (β̂ᵢ−β*)ᵀ Wᵢ (β̂ᵢ−β*)`.
        """
        return self._xp.sum(self.chi_squared_map.array)

    @property
    def noise_normalization(self) -> float:
        """
        `noise_norm = Σᵢ log((2π)² / det Wᵢ)`.
        """
        w11, w12, w21, w22 = precision_tensor_components_from(self, self.weighting)
        det_w = w11 * w22 - w12 * w21
        return self._xp.sum(self._xp.log((2.0 * np.pi) ** 2.0 / det_w))

    @property
    def marginalization_term(self) -> float:
        """
        The analytic-marginalization contribution to the log likelihood from integrating out the (flat-prior)
        source-plane centre: `-0.5 * log(det(Σᵢ Wᵢ) / (2π)²)` — the exact 2-D Gaussian integral over the
        centre contributes `(2π)^{d/2} / sqrt(det)` with `d = 2`.
        """
        w11, w12, w21, w22 = precision_tensor_components_from(self, self.weighting)
        sum_w11 = self._xp.sum(w11)
        sum_w12 = self._xp.sum(w12)
        sum_w21 = self._xp.sum(w21)
        sum_w22 = self._xp.sum(w22)
        det_sum_w = sum_w11 * sum_w22 - sum_w12 * sum_w21
        return -0.5 * self._xp.log(det_sum_w / (2.0 * np.pi) ** 2.0)

    @property
    def log_likelihood(self) -> float:
        """
        `log_likelihood = -0.5*(χ² + noise_norm) - 0.5*log(det(Σᵢ Wᵢ) / (2π)²)`.
        """
        return (
            -0.5 * (self.chi_squared + self.noise_normalization)
            + self.marginalization_term
        )
