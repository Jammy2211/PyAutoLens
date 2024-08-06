from typing import Optional, Tuple

import autoarray as aa
import autogalaxy as ag

from autolens.point.solver import PointSolver
from autolens.lens.tracer import Tracer

from autolens import exc


class AbstractFitPositionsImagePair:

    def __init__(
        self,
        name: str,
        positions: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: PointSolver,
        profile: Optional[ag.ps.Point] = None,
    ):
        """
        A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        ----------
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        super().__init__(dataset=positions)

        self.name = name
        self._noise_map = noise_map
        self.tracer = tracer
        self.solver = solver

        self.profile = (
            tracer.extract_profile(profile_name=name) if profile is None else profile
        )

        if self.profile is None:
            raise exc.PointExtractionException(
                f"For the point-source named {name} there was no matching point source profile "
                f"in the tracer (make sure your tracer's point source name is the same the dataset name."
            )

    @property
    def source_plane_coordinate(self) -> Tuple[float, float]:
        """
        Returns the centre of the point-source in the source-plane, which is used when computing the model
        image-plane positions from the tracer.

        Returns
        -------
        The (y,x) arc-second coordinates of the point-source in the source-plane.
        """
        return self.profile.centre

    @property
    def source_plane_index(self) -> int:
        """
        Returns the integer plane index containing the point source galaxy, which is used when computing the model
        image-plane positions from the tracer.

        This index is used to ensure that if multi-plane tracing is used when solving the model image-plane positions,
        the correct source-plane is used to compute the model positions whilst accounting for multi-plane lensing.

        Returns
        -------
        The index of the plane containing the point-source galaxy.
        """
        return self.tracer.extract_plane_index_of_profile(profile_name=self.name)

    @property
    def source_plane_redshift(self) -> float:
        """
        Returns the redshift of the plane containing the point source galaxy, which is used when computing the model
        image-plane positions from the tracer.

        This redshift is used to ensure that if multi-plane tracing is used when solving the model image-plane
        positions, the correct source-plane is used to compute the model positions whilst accounting for multi-plane
        lensing.

        Returns
        -------
        The redshift of the plane containing the point-source galaxy.
        """
        return self.tracer.planes[self.source_plane_index].redshift