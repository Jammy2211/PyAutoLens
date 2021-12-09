from functools import partial
import numba
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.point.point_dataset import PointDict
from autolens.point.point_dataset import PointDataset
from autolens.point.point_solver import PointSolver
from autolens.lens.ray_tracing import Tracer

from autolens import exc


class FitPointDict(dict):
    def __init__(
        self, point_dict: PointDict, tracer: Tracer, point_solver: PointSolver
    ):
        """
        A fit to a point source dataset, which is stored as a dictionary containing the fit of every data point in a
        entire point-source dataset dictionary.

        This dictionary uses the `name` of the `PointDataset` to act as the key of every entry of the dictionary,
        making it straight forward to access the attributes based on the dataset name.

        Parameters
        ----------
        point_dict
            A dictionary of all point-source datasets that are to be fitted.

        Returns
        -------
        Dict
            A dictionary where the keys are the `name` entries of each dataset in the `PointDict` and the values
            are the corresponding fits to the `PointDataset` it contained.
        """

        self.tracer = tracer

        super().__init__()

        for key, point_dataset in point_dict.items():

            self[key] = FitPointDataset(
                point_dataset=point_dataset, tracer=tracer, point_solver=point_solver
            )

    @property
    def log_likelihood(self) -> float:
        return sum(fit.log_likelihood for fit in self.values())


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


class FitPositionsImage(aa.FitData):
    def __init__(
        self,
        name: str,
        positions: aa.Grid2DIrregular,
        noise_map: aa.ValuesIrregular,
        tracer: Tracer,
        point_solver: PointSolver,
        point_profile: Optional[ag.ps.Point] = None,
    ):
        """
        A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        self.name = name

        if point_profile is None:
            point_profile = tracer.extract_profile(profile_name=name)

        self.point_profile = point_profile

        self.point_solver = point_solver

        if self.point_profile is None:
            raise exc.PointExtractionException(
                f"For the point-source named {name} there was no matching point source profile "
                f"in the tracer (make sure your tracer's point source name is the same the dataset name."
            )

        self.source_plane_coordinate = self.point_profile.centre

        if len(tracer.planes) > 2:
            upper_plane_index = tracer.extract_plane_index_of_profile(profile_name=name)
        else:
            upper_plane_index = None

        model_positions = point_solver.solve(
            lensing_obj=tracer,
            source_plane_coordinate=self.source_plane_coordinate,
            upper_plane_index=upper_plane_index,
        )

        model_positions = model_positions.grid_of_closest_from(grid_pair=positions)

        super().__init__(
            data=positions,
            noise_map=noise_map,
            model_data=model_positions,
            mask=None,
            inversion=None,
        )

    @property
    def positions(self) -> aa.Grid2DIrregular:
        return self.data

    @property
    def model_positions(self) -> aa.Grid2DIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ValuesIrregular:

        residual_positions = self.positions - self.model_positions

        return residual_positions.distances_to_coordinate(coordinate=(0.0, 0.0))


class FitPositionsSource(aa.FitData):
    def __init__(
        self,
        name: str,
        positions: aa.Grid2DIrregular,
        noise_map: aa.ValuesIrregular,
        tracer: Tracer,
        point_profile: Optional[ag.ps.Point] = None,
    ):
        """
        A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        self.name = name

        if point_profile is None:
            point_profile = tracer.extract_profile(profile_name=name)

        self.point_profile = point_profile

        if self.point_profile is None:
            raise exc.PointExtractionException(
                f"For the point-source named {name} there was no matching point source profile "
                f"in the tracer (make sure your tracer's point source name is the same the dataset name."
            )

        self.source_plane_coordinate = self.point_profile.centre

        if len(tracer.planes) <= 2:

            deflections = tracer.deflections_yx_2d_from(grid=positions)

        else:

            upper_plane_index = tracer.extract_plane_index_of_profile(profile_name=name)

            deflections = tracer.deflections_between_planes_from(
                grid=positions, plane_i=0, plane_j=upper_plane_index
            )

        model_positions = positions.grid_via_deflection_grid_from(
            deflection_grid=deflections
        )

        super().__init__(
            data=positions,
            noise_map=noise_map,
            model_data=model_positions,
            mask=None,
            inversion=None,
        )

    @property
    def positions(self) -> aa.Grid2DIrregular:
        return self.data

    @property
    def model_positions(self) -> aa.Grid2DIrregular:
        return self.model_data

    @property
    def residual_map(self) -> aa.ValuesIrregular:

        return self.model_positions.distances_to_coordinate(
            coordinate=self.source_plane_coordinate
        )


class FitFluxes(aa.FitData):
    def __init__(
        self,
        name: str,
        fluxes: aa.ValuesIrregular,
        noise_map: aa.ValuesIrregular,
        positions: aa.Grid2DIrregular,
        tracer: Tracer,
        point_profile: Optional[ag.ps.Point] = None,
    ):

        self.tracer = tracer

        self.name = name
        self.positions = positions

        if point_profile is None:
            point_profile = tracer.extract_profile(profile_name=name)

        self.point_profile = point_profile

        if self.point_profile is None:
            raise exc.PointExtractionException(
                f"For the point-source named {name} there was no matching point source profile "
                f"in the tracer (make sure your tracer's point source name is the same the dataset name."
            )

        elif not hasattr(self.point_profile, "flux"):
            raise exc.PointExtractionException(
                f"For the point-source named {name} the extracted point source was the "
                f"class {self.point_profile.__class__.__name__} and therefore does "
                f"not contain a flux component."
            )

        if len(tracer.planes) > 2:
            upper_plane_index = tracer.extract_plane_index_of_profile(profile_name=name)
            deflections_func = partial(
                tracer.deflections_between_planes_from,
                plane_i=0,
                plane_j=upper_plane_index,
            )
        else:
            deflections_func = tracer.deflections_yx_2d_from

        self.magnifications = abs(
            self.tracer.magnification_2d_via_hessian_from(
                grid=positions, deflections_func=deflections_func
            )
        )

        model_fluxes = aa.ValuesIrregular(
            values=[
                magnification * self.point_profile.flux
                for magnification in self.magnifications
            ]
        )

        super().__init__(
            data=fluxes,
            noise_map=noise_map,
            model_data=model_fluxes,
            mask=None,
            inversion=None,
        )

    @property
    def fluxes(self) -> aa.ValuesIrregular:
        return self.data

    @property
    def model_fluxes(self) -> aa.ValuesIrregular:
        return self.model_data


class AbstractFitPositionsSourcePlane:
    def __init__(
        self,
        positions: aa.Grid2DIrregular,
        noise_map: aa.ValuesIrregular,
        tracer: Tracer,
    ):
        """
        Given a positions dataset, which is a list of positions with names that associated them to model source
        galaxies, use a `Tracer` to determine the traced coordinate positions in the source-plane.

        Different children of this abstract class are available which use the traced coordinates to define a chi-squared
        value in different ways.

        Parameters
        -----------
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of named positions which the log_likelihood is computed using. Positions
            are paired to galaxies in the `Tracer` using their names.
        tracer : Tracer
            The object that defines the ray-tracing of the strong lens system of galaxies.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """
        self.positions = positions
        self.noise_map = noise_map
        self.source_plane_positions = tracer.traced_grid_list_from(grid=positions)[-1]

    @property
    def furthest_separations_of_source_plane_positions(self) -> aa.ValuesIrregular:
        """
        Returns the furthest distance of every source-plane (y,x) coordinate to the other source-plane (y,x)
        coordinates.

        For example, for the following source-plane positions:

        source_plane_positions = [[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)]

        The returned furthest distances are:

        source_plane_positions = [3.0, 2.0, 3.0]

        Returns
        -------
        aa.ValuesIrregular
            The further distances of every set of grouped source-plane coordinates the other source-plane coordinates
            that it is grouped with.
        """
        return self.source_plane_positions.furthest_distances_to_other_coordinates

    @property
    def max_separation_of_source_plane_positions(self) -> float:
        return max(self.furthest_separations_of_source_plane_positions)

    def max_separation_within_threshold(self, threshold) -> bool:
        return self.max_separation_of_source_plane_positions <= threshold


class FitPositionsSourceMaxSeparation(AbstractFitPositionsSourcePlane):
    def __init__(
        self,
        positions: aa.Grid2DIrregular,
        noise_map: Optional[aa.ValuesIrregular],
        tracer: Tracer,
    ):
        """A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """
        super().__init__(positions=positions, noise_map=noise_map, tracer=tracer)
