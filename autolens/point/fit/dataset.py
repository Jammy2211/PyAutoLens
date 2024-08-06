from typing import Callable, Dict, Optional


from autolens.point.dataset import PointDataset
from autolens.point.solver import PointSolver
from autolens.point.fit.fluxes import FitFluxes
from autolens.lens.tracer import Tracer

from autolens.point.fit.positions.image.pair import FitPositionsImagePair
from autolens import exc

try:
    import numba

    NumbaException = numba.errors.TypingError
except ModuleNotFoundError:
    NumbaException = AttributeError


class FitPointDataset:
    def __init__(
        self,
        dataset: PointDataset,
        tracer: Tracer,
        solver: PointSolver,
        fit_positions_cls = FitPositionsImagePair,
        run_time_dict: Optional[Dict] = None,
    ):
        self.dataset = dataset
        self.tracer = tracer
        self.solver = solver
        self.run_time_dict = run_time_dict

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
            )
        except exc.PointExtractionException:
            self.positions = None

        try:
            self.flux = FitFluxes(
                name=dataset.name,
                data=dataset.fluxes,
                noise_map=dataset.fluxes_noise_map,
                positions=dataset.positions,
                tracer=tracer,
            )

        except exc.PointExtractionException:
            self.flux = None

    @property
    def log_likelihood(self) -> float:
        log_likelihood_positions = (
            self.positions.log_likelihood if self.positions is not None else 0.0
        )
        log_likelihood_flux = self.flux.log_likelihood if self.flux is not None else 0.0

        return log_likelihood_positions + log_likelihood_flux

    @property
    def figure_of_merit(self) -> float:
        return self.log_likelihood
