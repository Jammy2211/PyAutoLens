from typing import Dict, Optional

import autogalaxy as ag

from autolens.point.point_dataset import PointDataset
from autolens.point.solver import PointSolver
from autolens.point.fit_point.fluxes import FitFluxes
from autolens.point.fit_point.positions.image_pair import FitPositionsImagePair
from autolens.point.fit_point.positions_source import FitPositionsSource
from autolens.lens.tracer import Tracer

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
        tracer : Tracer,
        point_solver: PointSolver,
        run_time_dict: Optional[Dict] = None,
    ):
        self.dataset = dataset
        self.tracer = tracer
        self.point_solver = point_solver
        self.run_time_dict = run_time_dict

        profile = self.tracer.extract_profile(profile_name=dataset.name)

        try:
            if isinstance(profile, ag.ps.PointSourceChi):
                self.positions = FitPositionsSource(
                    name=dataset.name,
                    positions=dataset.positions,
                    noise_map=dataset.positions_noise_map,
                    tracer=tracer,
                    profile=profile,
                )

            else:
                self.positions = FitPositionsImagePair(
                    name=dataset.name,
                    positions=dataset.positions,
                    noise_map=dataset.positions_noise_map,
                    tracer=tracer,
                    solver=point_solver,
                    profile=profile,
                )

        except exc.PointExtractionException:
            self.positions = None
        except (AttributeError, NumbaException) as e:
            raise exc.FitException from e

        try:
            self.flux = FitFluxes(
                name=dataset.name,
                fluxes=dataset.fluxes,
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
