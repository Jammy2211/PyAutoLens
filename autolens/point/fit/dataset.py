import numpy as np

from autolens.point.dataset import PointDataset
from autolens.point.solver import PointSolver
from autolens.point.fit.fluxes import FitFluxes
from autolens.point.fit.times_delays import FitTimeDelays
from autolens.lens.tracer import Tracer

from autolens.point.fit.positions.image.pair import FitPositionsImagePair
from autolens import exc


class FitPointDataset:
    def __init__(
        self,
        dataset: PointDataset,
        tracer: Tracer,
        solver: PointSolver,
        fit_positions_cls=FitPositionsImagePair,
        xp=np,
    ):
        """
        Fits a point source dataset using a `Tracer` object, where the following components of the point source data
        may be fitted:

        - The positions of the point source in the image-plane, where the chi-squared could be defined as an image-plane
          or source-plane chi-squared.

        - The fluxes of the point source, which use the magnification of the point source to compute the fluxes in the
          image-plane.

        - The time delays of the point source in delays, which use the tracer to compute the model time delays
          at the image-plane positions of the point source in the dataset.

        The fit may use one or combinations of the above components to compute the log likelihood, depending on what
        components are available in the point source dataset and the model point source profiles input. For example:

        - The `ps.Point` object has a `centre` but does not have a flux, so the fluxes are not fitted, meaning only
          positions are fitted.

        - The `ps.PointFlux` object has a `centre` and a flux, therefore both the positions and fluxes are fitted.

        The fit performs the following steps:

        1) Fit the positions of the point source dataset using the input `fit_positions_cls` object, which could be an
           image-plane or source-plane chi-squared.

        2) Fit the fluxes of the point source dataset using the `FitFluxes` object, where the object type may be
          extended in the future to support different types of point source profiles.

        3) Fits the time delays of the point source dataset using the `FitTimeDelays` object, which is an image-plane
           evaluation of the time delays at the image-plane positions of the point source in the dataset.

        Point source fitting uses name pairing, whereby the `name` of the `Point` object is paired to the name of the
        point source dataset to ensure that point source datasets are fitted to the correct point source.

        When performing a `model-fit`via an `AnalysisPoint` object the `figure_of_merit` of this object
        is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The point source dataset which is fitted.
        tracer
            The tracer of galaxies whose point source profile are used to fit the positions.
        solver
            Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
            triangles to and from the source-plane.
        fit_positions_cls
            The class used to fit the positions of the point source dataset, which could be an image-plane or
            source-plane chi-squared.
        profile
            Manually input the profile of the point source, which is used instead of the one extracted from the
            tracer via name pairing if that profile is not found.
        """
        self.dataset = dataset
        self.tracer = tracer
        self.solver = solver

        profile = self.tracer.extract_profile(profile_name=dataset.name)

        self.fit_positions_cls = fit_positions_cls

        try:
            self.positions = self.fit_positions_cls(
                name=dataset.name,
                data=dataset.positions,
                noise_map=dataset.positions_noise_map,
                tracer=tracer,
                solver=solver,
                profile=profile,
                xp=xp,
            )
        except exc.PointExtractionException:
            self.positions = None

        try:
            if dataset.fluxes is not None:
                self.flux = FitFluxes(
                    name=dataset.name,
                    data=dataset.fluxes,
                    noise_map=dataset.fluxes_noise_map,
                    positions=dataset.positions,
                    tracer=tracer,
                    xp=xp,
                )
            else:
                self.flux = None

        except exc.PointExtractionException:
            self.flux = None

        try:
            if dataset.time_delays is not None:
                self.time_delays = FitTimeDelays(
                    name=dataset.name,
                    data=dataset.time_delays,
                    noise_map=dataset.time_delays_noise_map,
                    positions=dataset.positions,
                    tracer=tracer,
                    xp=xp,
                )
            else:
                self.time_delays = None
        except exc.PointExtractionException:
            self.time_delays = None

        self._xp = xp

    @property
    def model_obj(self):
        return self.tracer

    @property
    def log_likelihood(self) -> float:
        """
        Returns the overall `log_likelihood` of the point source dataset, which is the sum of the log likelihoods of
        each individual component of the point source dataset that is fitted (e.g. positions, fluxes, time delays).
        """
        log_likelihood_positions = (
            self.positions.log_likelihood if self.positions is not None else 0.0
        )
        log_likelihood_flux = self.flux.log_likelihood if self.flux is not None else 0.0
        log_likelihood_time_delays = (
            self.time_delays.log_likelihood if self.time_delays is not None else 0.0
        )

        return (
            log_likelihood_positions + log_likelihood_flux + log_likelihood_time_delays
        )

    @property
    def figure_of_merit(self) -> float:
        """
        The `figure_of_merit` of the point source dataset, which is the value the `AnalysisPoint` object calls to
        perform a model-fit.
        """
        return self.log_likelihood
