from abc import ABC
from astropy import cosmology as cosmo
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import autoarray as aa
import autogalaxy as ag

from autoconf.dictable import Dictable

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from

from autogalaxy.plane.plane import Plane
from autogalaxy.profiles.light_profiles.light_profiles_snr import LightProfileSNR

from autolens.analysis.preloads import Preloads

from autolens.lens import ray_tracing_util


class Tracer(ABC, ag.OperateImageGalaxies, ag.OperateDeflections, Dictable):
    def __init__(self, planes, cosmology, profiling_dict: Optional[Dict] = None):
        """
        Ray-tracer for a lens system with any number of planes.

        The redshift of these planes are specified by the redshits of the galaxies; there is a unique plane redshift \
        for every unique galaxy redshift (galaxies with identical redshifts are put in the same plane).

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid (see gridStack) which is used for ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid : grid_stacks.GridStack
            The image-plane grid which is traced. (includes the grid, sub-grid, blurring-grid, etc.).
        border : masks.GridBorder
            The border of the grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """
        self.planes = planes
        self.plane_redshifts = [plane.redshift for plane in planes]
        self.cosmology = cosmology

        self.profiling_dict = profiling_dict

    @classmethod
    def from_galaxies(
        cls, galaxies, cosmology=cosmo.Planck15, profiling_dict: Optional[Dict] = None
    ):

        planes = ag.util.plane.planes_via_galaxies_from(
            galaxies=galaxies, profiling_dict=profiling_dict
        )

        return Tracer(planes=planes, cosmology=cosmology, profiling_dict=profiling_dict)

    @classmethod
    def sliced_tracer_from(
        cls,
        lens_galaxies,
        line_of_sight_galaxies,
        source_galaxies,
        planes_between_lenses,
        cosmology=cosmo.Planck15,
    ):

        """Ray-tracer for a lens system with any number of planes.

        The redshift of these planes are specified by the input parameters *lens_redshifts* and \
         *slices_between_main_planes*. Every galaxy is placed in its closest plane in redshift-space.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid (see gridStack) which is used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid : grid_stacks.GridStack
            The image-plane grid which is traced. (includes the grid, sub-grid, blurring-grid, etc.).
        planes_between_lenses : [int]
            The number of slices between each main plane. The first entry in this list determines the number of slices \
            between Earth (redshift 0.0) and main plane 0, the next between main planes 0 and 1, etc.
        border : masks.GridBorder
            The border of the grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        lens_redshifts = ag.util.plane.ordered_plane_redshifts_from(
            galaxies=lens_galaxies
        )

        plane_redshifts = ag.util.plane.ordered_plane_redshifts_with_slicing_from(
            lens_redshifts=lens_redshifts,
            planes_between_lenses=planes_between_lenses,
            source_plane_redshift=source_galaxies[0].redshift,
        )

        galaxies_in_planes = ag.util.plane.galaxies_in_redshift_ordered_planes_from(
            galaxies=lens_galaxies + line_of_sight_galaxies,
            plane_redshifts=plane_redshifts,
        )

        plane_redshifts.append(source_galaxies[0].redshift)
        galaxies_in_planes.append(source_galaxies)

        planes = []

        for plane_index in range(0, len(plane_redshifts)):
            planes.append(
                ag.Plane(
                    redshift=plane_redshifts[plane_index],
                    galaxies=galaxies_in_planes[plane_index],
                )
            )

        return Tracer(planes=planes, cosmology=cosmology)

    def dict(self) -> Dict:
        tracer_dict = super().dict()
        tracer_dict["cosmology"] = self.cosmology.name
        tracer_dict["planes"] = [plane.dict() for plane in self.planes]
        return tracer_dict

    @staticmethod
    def from_dict(cls_dict):
        cls_dict["cosmology"] = getattr(cosmo, cls_dict["cosmology"])
        cls_dict["planes"] = list(map(Plane.from_dict, cls_dict["planes"]))
        return Dictable.from_dict(cls_dict)

    @property
    def galaxies(self) -> List[ag.Galaxy]:
        return list([galaxy for plane in self.planes for galaxy in plane.galaxies])

    @property
    def total_planes(self) -> int:
        return len(self.plane_redshifts)

    @property
    def image_plane(self) -> Plane:
        return self.planes[0]

    @property
    def source_plane(self) -> Plane:
        return self.planes[-1]

    @property
    def all_planes_have_redshifts(self) -> bool:
        return None not in self.plane_redshifts

    def plane_with_galaxy(self, galaxy) -> Plane:
        return [plane for plane in self.planes if galaxy in plane.galaxies][0]

    @aa.profile_func
    def sparse_image_plane_grid_pg_list_from(
        self, grid: aa.type.Grid2DLike, settings_pixelization=aa.SettingsPixelization()
    ) -> List[List]:
        """
        Specific pixelizations, like the `VoronoiMagnification`, begin by determining what will become its the
        source-pixel centres by calculating them  in the image-plane. The `VoronoiBrightnessImage` pixelization
        performs a KMeans clustering.
        """

        sparse_image_plane_grid_list_of_planes = []

        for plane in self.planes:
            sparse_image_plane_grid_list = plane.sparse_image_plane_grid_list_from(
                grid=grid, settings_pixelization=settings_pixelization
            )
            sparse_image_plane_grid_list_of_planes.append(sparse_image_plane_grid_list)

        return sparse_image_plane_grid_list_of_planes

    @aa.grid_dec.grid_2d_to_structure_list
    def traced_grid_2d_list_from(
        self, grid: aa.type.Grid2DLike, plane_index_limit=None
    ) -> List[aa.type.Grid2DLike]:
        """
        Performs multi-plane ray tracing on a 2D grid of Cartesian (y,x) coordinates using the mass profiles of the
        galaxies and planes contained within the tracer.

        see `autolens.lens.ray_tracing.ray_tracing_util.traced_grid_2d_list_from()` for a full description of the
        calculation.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates on which multi-plane ray-tracing calculations are performed.
        plane_index_limit
            The integer index of the last plane which is used to perform ray-tracing, all planes with an index above
            this value are omitted.

        Returns
        -------
        traced_grid_list
            A list of 2D (y,x) grids each of which are the input grid ray-traced to a redshift of the input list of planes.
        """

        return ray_tracing_util.traced_grid_2d_list_from(
            planes=self.planes,
            grid=grid,
            cosmology=self.cosmology,
            plane_index_limit=plane_index_limit,
        )

    @aa.profile_func
    def traced_sparse_grid_pg_list_from(
        self,
        grid: aa.type.Grid2DLike,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=Preloads(),
    ) -> Tuple[List[List], List[List]]:
        """
        Ray-trace the sparse image plane grid used to define the source-pixel centres by calculating the deflection
        angles at (y,x) coordinate on the grid from the galaxy mass profiles and then ray-trace them from the
        image-plane to the source plane.
        """
        if (
            preloads.sparse_image_plane_grid_pg_list is None
            or settings_pixelization.is_stochastic
        ):

            sparse_image_plane_grid_pg_list = self.sparse_image_plane_grid_pg_list_from(
                grid=grid, settings_pixelization=settings_pixelization
            )

        else:

            sparse_image_plane_grid_pg_list = preloads.sparse_image_plane_grid_pg_list

        traced_sparse_grid_pg_list = []

        for (plane_index, plane) in enumerate(self.planes):

            if sparse_image_plane_grid_pg_list[plane_index] is None:
                traced_sparse_grid_pg_list.append(None)
            else:

                traced_sparse_grids_list = []

                for sparse_image_plane_grid in sparse_image_plane_grid_pg_list[
                    plane_index
                ]:

                    try:
                        traced_sparse_grids_list.append(
                            self.traced_grid_2d_list_from(grid=sparse_image_plane_grid)[
                                plane_index
                            ]
                        )
                    except AttributeError:
                        traced_sparse_grids_list.append(None)

                traced_sparse_grid_pg_list.append(traced_sparse_grids_list)

        return traced_sparse_grid_pg_list, sparse_image_plane_grid_pg_list

    def grid_2d_at_redshift_from(
        self, grid: aa.type.Grid2DLike, redshift: float
    ) -> aa.type.Grid2DLike:
        """
        For an input grid of (y,x) arc-second image-plane coordinates, ray-trace the coordinates to any redshift in \
        the strong lens configuration.

        This is performed using multi-plane ray-tracing and the existing redshifts and planes of the tracer. However, \
        any redshift can be input even if a plane does not exist there, including redshifts before the first plane \
        of the lens system.

        Parameters
        ----------
        grid : ndsrray or aa.Grid2D
            The image-plane grid which is traced to the redshift.
        redshift
            The redshift the image-plane grid is traced to.
        """
        return ray_tracing_util.grid_2d_at_redshift_from(
            redshift=redshift,
            galaxies=self.galaxies,
            grid=grid,
            cosmology=self.cosmology,
        )

    @property
    def has_light_profile(self) -> bool:
        return any(list(map(lambda plane: plane.has_light_profile, self.planes)))

    @aa.grid_dec.grid_2d_to_structure
    @aa.profile_func
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        return sum(self.image_2d_list_from(grid=grid))

    @aa.grid_dec.grid_2d_to_structure_list
    def image_2d_list_from(self, grid: aa.type.Grid2DLike) -> List[aa.Array2D]:

        traced_grid_list = self.traced_grid_2d_list_from(
            grid=grid, plane_index_limit=self.upper_plane_index_with_light_profile
        )

        image_2d_list = [
            self.planes[plane_index].image_2d_from(grid=traced_grid_list[plane_index])
            for plane_index in range(len(traced_grid_list))
        ]

        if self.upper_plane_index_with_light_profile < self.total_planes - 1:
            for plane_index in range(
                self.upper_plane_index_with_light_profile, self.total_planes - 1
            ):
                image_2d_list.append(np.zeros(shape=image_2d_list[0].shape))

        return image_2d_list

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike
    ) -> {ag.Galaxy: np.ndarray}:
        """
        Returns a dictionary associating every `Galaxy` object in the `Tracer` with its corresponding 2D image, using
        the instance of each galaxy as the dictionary keys.

        This object is used for hyper-features, which use the image of each galaxy in a model-fit in order to
        adapt quantities like a pixelization or regularization scheme to the surface brightness of the galaxies being
        fitted.

        By inheriting from `OperateImageGalaxies` functions which apply operations of this dictionary are accessible,
        for example convolving every image with a PSF or applying a Fourier transform to create a galaxy-visibilities
        dictionary.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.

        Returns
        -------
        A dictionary associated every galaxy in the tracer with its corresponding 2D image.
        """

        galaxy_image_dict = dict()

        traced_grid_list = self.traced_grid_2d_list_from(grid=grid)

        for (plane_index, plane) in enumerate(self.planes):
            images_of_galaxies = plane.image_2d_list_from(
                grid=traced_grid_list[plane_index]
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_image_dict[galaxy] = images_of_galaxies[galaxy_index]

        return galaxy_image_dict

    @property
    def has_mass_profile(self) -> bool:
        return any(list(map(lambda plane: plane.has_mass_profile, self.planes)))

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:
        if self.total_planes > 1:
            return self.deflections_between_planes_from(grid=grid)
        return self.planes[0].deflections_yx_2d_from(grid=grid)

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_of_planes_summed_from(
        self, grid: aa.type.Grid2DLike
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:
        return sum([plane.deflections_yx_2d_from(grid=grid) for plane in self.planes])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_between_planes_from(
        self, grid: aa.type.Grid2DLike, plane_i=0, plane_j=-1
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:

        traced_grids_list = self.traced_grid_2d_list_from(grid=grid)

        return traced_grids_list[plane_i] - traced_grids_list[plane_j]

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        return sum([plane.convergence_2d_from(grid=grid) for plane in self.planes])

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        return sum([plane.potential_2d_from(grid=grid) for plane in self.planes])

    @property
    def has_pixelization(self) -> bool:
        return any(list(map(lambda plane: plane.has_pixelization, self.planes)))

    @property
    def has_regularization(self) -> bool:
        return any(list(map(lambda plane: plane.has_regularization, self.planes)))

    @aa.profile_func
    def traced_grid_2d_list_of_inversion_from(
        self, grid: aa.type.Grid2DLike
    ) -> List[aa.type.Grid2DLike]:
        return self.traced_grid_2d_list_from(grid=grid)

    @property
    def has_hyper_galaxy(self) -> bool:
        return any(list(map(lambda plane: plane.has_hyper_galaxy, self.planes)))

    @property
    def upper_plane_index_with_light_profile(self) -> int:
        return max(
            [
                plane_index if plane.has_light_profile else 0
                for (plane_index, plane) in enumerate(self.planes)
            ]
        )

    @property
    def mass_profile_list(self):
        return [
            plane.mass_profile_list for plane in self.planes if plane.has_mass_profile
        ]

    @property
    def plane_indexes_with_pixelizations(self):
        plane_indexes_with_inversions = [
            plane_index if plane.has_pixelization else None
            for (plane_index, plane) in enumerate(self.planes)
        ]
        return [
            plane_index
            for plane_index in plane_indexes_with_inversions
            if plane_index is not None
        ]

    @property
    def pixelization_list(self) -> List:
        return [
            galaxy.pixelization for galaxy in self.galaxies if galaxy.has_pixelization
        ]

    @property
    def pixelization_pg_list(self) -> List[List]:
        return [plane.pixelization_list for plane in self.planes]

    @property
    def regularization_list(self) -> List:
        return [
            galaxy.regularization
            for galaxy in self.galaxies
            if galaxy.has_regularization
        ]

    @property
    def regularization_pg_list(self) -> List[List]:
        return [plane.regularization_list for plane in self.planes]

    def linear_obj_list_from(
        self,
        grid: aa.type.Grid2DLike,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=Preloads(),
    ):

        linear_obj_list = []

        if preloads.traced_grids_of_planes_for_inversion is None:
            traced_grids_of_planes = self.traced_grid_2d_list_of_inversion_from(
                grid=grid
            )
        else:
            traced_grids_of_planes = preloads.traced_grids_of_planes_for_inversion

        if preloads.traced_sparse_grids_list_of_planes is None:
            traced_sparse_grids_list_of_planes, sparse_image_plane_grid_list = self.traced_sparse_grid_pg_list_from(
                grid=grid,
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )
        else:
            traced_sparse_grids_list_of_planes = (
                preloads.traced_sparse_grids_list_of_planes
            )
            sparse_image_plane_grid_list = preloads.sparse_image_plane_grid_list

        for (plane_index, plane) in enumerate(self.planes):

            if plane.has_pixelization:

                for mapper_index in range(
                    len(traced_sparse_grids_list_of_planes[plane_index])
                ):

                    mapper = plane.mapper_from(
                        source_grid_slim=traced_grids_of_planes[plane_index],
                        source_pixelization_grid=traced_sparse_grids_list_of_planes[
                            plane_index
                        ][mapper_index],
                        data_pixelization_grid=sparse_image_plane_grid_list[
                            plane_index
                        ][mapper_index],
                        pixelization=self.pixelization_pg_list[plane_index][
                            mapper_index
                        ],
                        hyper_galaxy_image=self.hyper_galaxy_image_pg_list[plane_index][
                            mapper_index
                        ],
                        settings_pixelization=settings_pixelization,
                        preloads=preloads,
                    )
                    linear_obj_list.append(mapper)

        return linear_obj_list

    def inversion_imaging_from(
        self,
        grid,
        image,
        noise_map,
        convolver,
        w_tilde,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=Preloads(),
    ):

        if preloads.linear_obj_list is None:

            linear_obj_list = self.linear_obj_list_from(
                grid=grid,
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )

        else:

            linear_obj_list = preloads.linear_obj_list

        return inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.regularization_list,
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.profiling_dict,
        )

    def inversion_interferometer_from(
        self,
        grid,
        visibilities,
        noise_map,
        transformer,
        w_tilde,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=Preloads(),
    ):

        if preloads.linear_obj_list is None:

            linear_obj_list = self.linear_obj_list_from(
                grid=grid,
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )

        else:

            linear_obj_list = preloads.linear_obj_list

        return inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.regularization_list,
            settings=settings_inversion,
            profiling_dict=self.profiling_dict,
        )

    @property
    def hyper_galaxy_image_pg_list(self) -> List[List]:
        return [
            plane.hyper_galaxies_with_pixelization_image_list for plane in self.planes
        ]

    def hyper_noise_map_from(self, noise_map: aa.Array2D) -> aa.Array2D:
        return sum(self.hyper_noise_map_list_from(noise_map=noise_map))

    def hyper_noise_map_list_from(self, noise_map: aa.Array2D) -> List[aa.Array2D]:
        return [
            plane.hyper_noise_map_from(noise_map=noise_map) for plane in self.planes
        ]

    @property
    def contribution_map(self) -> Optional[aa.Array2D]:

        contribution_map_list = self.contribution_map_list

        contribution_map_list = [i for i in contribution_map_list if i is not None]

        if contribution_map_list:
            return sum(contribution_map_list)
        else:
            return None

    @property
    def contribution_map_list(self) -> List[aa.Array2D]:

        contribution_map_list = []

        for plane in self.planes:

            if plane.contribution_map is not None:

                contribution_map_list.append(plane.contribution_map)

            else:

                contribution_map_list.append(None)

        return contribution_map_list

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class in the tracer as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if a tracer has an image-plane with a galaxy with two light profiles, the following:

        `tracer.extract_attribute(cls=LightProfile, name="axis_ratio")`

        would return:

        ValuesIrregular(values=[axis_ratio_0, axis_ratio_1])

        If the image plane has has two galaxies with two mass profiles and the source plane another galaxy with a
        mass profile, the following:

        `tracer.extract_attribute(cls=MassProfile, name="centre")`

        would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """

        def extract(value, name):

            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for galaxy in self.galaxies
            for value in galaxy.__dict__.values()
            if isinstance(value, cls)
        ]

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ValuesIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(grid=attributes)

    def extract_attributes_of_planes(self, cls, attr_name, filter_nones=False):
        """
        Returns an attribute of a class in the tracer as a list of `ValueIrregular` or `Grid2DIrregular` objects, where
        the indexes of the list correspond to the tracer's planes.

        For example, if a tracer has an image-plane with a galaxy with a light profile and a source-plane with a galaxy
        with a light profile, the following:

        `tracer.extract_attributes_of_planes(cls=LightProfile, name="axis_ratio")`

        would return:

        [ValuesIrregular(values=[axis_ratio_0]), ValuesIrregular(values=[axis_ratio_1])]

        If the image plane has two galaxies with a mass profile each and the source plane another galaxy with a
        mass profile, the following:

        `tracer.extract_attributes_of_planes(cls=MassProfile, name="centres")`

        would return:

        [
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0)]),
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)])
        ]

        If a Profile does not have a certain entry, it is replaced with a None, although the nones can be removed
        by setting `filter_nones=True`.

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """
        if filter_nones:

            return [
                plane.extract_attribute(cls=cls, attr_name=attr_name)
                for plane in self.planes
                if plane.extract_attribute(cls=cls, attr_name=attr_name) is not None
            ]

        else:

            return [
                plane.extract_attribute(cls=cls, attr_name=attr_name)
                for plane in self.planes
            ]

    def extract_attributes_of_galaxies(self, cls, attr_name, filter_nones=False):
        """
        Returns an attribute of a class in the tracer as a list of `ValueIrregular` or `Grid2DIrregular` objects, where
        the indexes of the list correspond to the tracer's galaxies. If a plane has multiple galaxies they are split
        into separate indexes int he list.

        For example, if a tracer has an image-plane with a galaxy with a light profile and a source-plane with a galaxy
        with a light profile, the following:

        `tracer.extract_attributes_of_galaxies(cls=LightProfile, name="axis_ratio")`

        would return:

        [ValuesIrregular(values=[axis_ratio_0]), ValuesIrregular(values=[axis_ratio_1])]

        If the image plane has two galaxies with a mass profile each and the source plane another galaxy with a
        mass profile, the following:

        `tracer.extract_attributes_of_galaxies(cls=MassProfile, name="centres")`

        would return:

        [
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0)]),
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0)])
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0)])
        ]

        If the first galaxy in the image plane in the example above had two mass profiles as well as the galaxy in the
        source plane it would return:

                [
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)]),
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0)])
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0, (centre_y_1, centre_x_1))])
        ]

        If a Profile does not have a certain entry, it is replaced with a None, although the nones can be removed
        by setting `filter_nones=True`.

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """
        if filter_nones:

            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
                if galaxy.extract_attribute(cls=cls, attr_name=attr_name) is not None
            ]

        else:

            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
            ]

    def extract_profile(self, profile_name):
        """
        Returns a `LightProfile`, `MassProfile` or `Point` from the `Tracer` using the name of that component.

        For example, if a tracer has two galaxies, `lens` and `source` with `LightProfile`'s name `light_0` and
        `light_1`, the following:

        `tracer.extract_profile(profile_name="light_1")`

        Would return the `LightProfile` of the source plane.
        """
        for galaxy in self.galaxies:
            try:
                return galaxy.__dict__[profile_name]
            except KeyError:
                pass

    def extract_plane_index_of_profile(self, profile_name):
        """
        Returns the plane index of a  LightProfile`, `MassProfile` or `Point` from the `Tracer` using the name
        of that component.

        For example, if a tracer has two galaxies, `lens` and `source` with `LightProfile`'s name `light_0` and
        `light_1`, the following:

        `tracer.extract_profile(profile_name="light_1")`

        Would return `plane_index=1` given the profile is in the source plane.
        """
        for plane_index, plane in enumerate(self.planes):
            for galaxy in plane.galaxies:
                if profile_name in galaxy.__dict__:
                    return plane_index

    def set_snr_of_snr_light_profiles(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
    ):

        grid = aa.Grid2D.uniform(
            shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, sub_size=1
        )

        traced_grids_of_planes = self.traced_grid_2d_list_from(grid=grid)

        for plane_index, plane in enumerate(self.planes):
            for galaxy in plane.galaxies:
                for light_profile in galaxy.light_profile_list:
                    if isinstance(light_profile, LightProfileSNR):
                        light_profile.set_intensity_from(
                            grid=traced_grids_of_planes[plane_index],
                            exposure_time=exposure_time,
                            background_sky_level=background_sky_level,
                        )

    @aa.profile_func
    def convolve_via_convolver(self, image, blurring_image, convolver):

        return convolver.convolve_image(image=image, blurring_image=blurring_image)
