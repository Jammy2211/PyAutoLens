import numba

import autogalaxy as ag

from autolens.point.point_dataset import PointDataset
from autolens.point.point_solver import PointSolver
from autolens.point.fit_point.fluxes import FitFluxes
from autolens.point.fit_point.positions_image import FitPositionsImage
from autolens.point.fit_point.positions_source import FitPositionsSource
from autolens.lens.ray_tracing import Tracer

from autolens import exc


class FitPointDataset:
    def __init__(
        self, point_dataset: PointDataset, tracer: Tracer, point_solver: PointSolver
    ):

        self.point_dataset = point_dataset

        point_profile = tracer.extract_profile(profile_name=point_dataset.name)

        try:

            if isinstance(point_profile, ag.ps.PointSourceChi):

                self.positions = FitPositionsSource(
                    name=point_dataset.name,
                    positions=point_dataset.positions,
                    noise_map=point_dataset.positions_noise_map,
                    tracer=tracer,
                    point_profile=point_profile,
                )

            else:

                self.positions = FitPositionsImage(
                    name=point_dataset.name,
                    positions=point_dataset.positions,
                    noise_map=point_dataset.positions_noise_map,
                    point_solver=point_solver,
                    tracer=tracer,
                    point_profile=point_profile,
                )

        except exc.PointExtractionException:
            self.positions = None
        except (AttributeError, numba.errors.TypingError) as e:
            raise exc.FitException from e

        try:

            self.flux = FitFluxes(
                name=point_dataset.name,
                fluxes=point_dataset.fluxes,
                noise_map=point_dataset.fluxes_noise_map,
                positions=point_dataset.positions,
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
