from abc import ABC
import numpy as np
from scipy.interpolate import griddata
from typing import Dict, List, Optional, Type, Union

import autoarray as aa
import autogalaxy as ag

from autogalaxy.plane.plane import Plane
from autogalaxy.profiles.light.snr import LightProfileSNR

from autolens.lens import ray_tracing_util


class Tracer(ABC, ag.OperateImageGalaxies, ag.OperateDeflections):
    def __init__(
        self,
        planes,
        cosmology: ag.cosmo.LensingCosmology,
        run_time_dict: Optional[Dict] = None,
    ):
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

        self.run_time_dict = run_time_dict

    @classmethod
    def from_galaxies(
        cls,
        galaxies,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
        run_time_dict: Optional[Dict] = None,
    ):
        planes = ag.util.plane.planes_via_galaxies_from(
            galaxies=galaxies, run_time_dict=run_time_dict
        )

        return cls(planes=planes, cosmology=cosmology, run_time_dict=run_time_dict)

    @classmethod
    def sliced_tracer_from(
        cls,
        lens_galaxies,
        line_of_sight_galaxies,
        source_galaxies,
        planes_between_lenses,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
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

    @property
    def galaxies(self) -> List[ag.Galaxy]:
        return list([galaxy for plane in self.planes for galaxy in plane.galaxies])

    def has(self, cls: Type) -> bool:
        return any(map(lambda plane: plane.has(cls=cls), self.planes))

    def cls_list_from(self, cls: Type) -> List:
        """
        Returns a list of objects in the tracer which are an instance of the input `cls`.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the tracer is returned.

        Returns
        -------
            The list of objects in the tracer that inherit from input `cls`.
        """
        cls_list = []

        for galaxy in self.galaxies:
            if galaxy.has(cls=cls):
                for cls_galaxy in galaxy.cls_list_from(cls=cls):
                    cls_list.append(cls_galaxy)

        return cls_list

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

    @aa.grid_dec.grid_2d_to_structure
    @aa.profile_func
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        return sum(self.image_2d_list_from(grid=grid, operated_only=operated_only))

    @aa.grid_dec.grid_2d_to_structure_list
    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        traced_grid_list = self.traced_grid_2d_list_from(
            grid=grid, plane_index_limit=self.upper_plane_index_with_light_profile
        )

        image_2d_list = [
            self.planes[plane_index].image_2d_from(
                grid=traced_grid_list[plane_index], operated_only=operated_only
            )
            for plane_index in range(len(traced_grid_list))
        ]

        if self.upper_plane_index_with_light_profile < self.total_planes - 1:
            for plane_index in range(
                self.upper_plane_index_with_light_profile, self.total_planes - 1
            ):
                image_2d_list.append(np.zeros(shape=image_2d_list[0].shape))

        return image_2d_list

    def image_2d_via_input_plane_image_from(
        self,
        grid: aa.type.Grid2DLike,
        plane_image: aa.Array2D,
        plane_index: int = -1,
        include_other_planes: bool = True,
    ) -> aa.Array2D:
        """
        Returns the lensed image of a plane or galaxy, where the input image is uniform and interpolated to compute
        the lensed image.

        The typical use case is inputting the image of an irregular galaxy in the source-plane (whose values are
        on a uniform array) and using this function computing the lensed image of this source galaxy.

        By default, this function computes the lensed image of the final plane, which is the source-plane, by using
        `plane_index=-1`. For multi-plane lens systems, the lensed image of any planes can be computed by setting
        `plane_index` to the index of the plane in the lens system.

        The emission of all other planes and galaxies can be included or omitted setting the `include_other_planes`
        bool. If there are multiple planes in a multi-plane lens system, the emission of the other planes are fully
        lensed.

        __Source Plane Interpolation__

        We use the scipy interpolation function `griddata` to create the lensed source galaxy image.

        In brief, we trace light rays to the source plane and calculate values based on where those light rays land in
        the source plane via interpolation.

        In more detail:

        - `points`: The 2D grid of (y,x) coordinates representing the location of every pixel of the source galaxy
          image in the source-plane, from which we are creating the lensed source image. These coordinates are the
          uniform source-plane grid computed after interpolating the irregular mesh the original source reconstruction
          used.

        - `values`: The intensity values of the source galaxy image which is used to create the lensed source image.
           These values are the flux values of the interpolated source galaxy image computed after interpolating the
           irregular mesh the original source reconstruction used.

        - `xi`: The image-plane grid ray traced to the source-plane. This evaluates the flux of each image-plane
          lensed source-pixel by ray-tracing it to the source-plane grid and computing its value by interpolating the
          source galaxy image.

        Parameters
        ----------
        grid
            The image-plane grid which is traced to the plane where the image is computed, where these values are
            used to perform the interpolation.
        plane_image
            The image of the plane or galaxy which is interpolated to compute the lensed image.
        plane_index
            The index of the plane the image is computed, where the default (-1) computes the image in the last plane
            and therefore the source-plane.

        Returns
        -------
        The lensed image of the plane or galaxy computed by interpolating its image to the image-plane.
        """

        plane_grid = aa.Grid2D.uniform(
            shape_native=plane_image.shape_native,
            pixel_scales=plane_image.pixel_scales,
            sub_size=plane_image.sub_size,
        )

        traced_grid = self.traced_grid_2d_list_from(
            grid=grid, plane_index_limit=plane_index
        )[plane_index]

        image = griddata(
            points=plane_grid,
            values=plane_image,
            xi=traced_grid,
            fill_value=0.0,
            method="linear",
        )

        if include_other_planes:
            image_list = self.image_2d_list_from(grid=grid, operated_only=False)

            if plane_index < 0:
                plane_index = self.total_planes + plane_index

            for plane_lp_index in range(self.total_planes):
                if plane_lp_index != plane_index:
                    image += image_list[plane_lp_index]

        return aa.Array2D(
            values=image,
            mask=grid.mask,
        )

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> Dict[ag.Galaxy, np.ndarray]:
        """
        Returns a dictionary associating every `Galaxy` object in the `Tracer` with its corresponding 2D image, using
        the instance of each galaxy as the dictionary keys.

        This object is used for adaptive-features, which use the image of each galaxy in a model-fit in order to
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

        galaxy_image_2d_dict = dict()

        traced_grid_list = self.traced_grid_2d_list_from(grid=grid)

        for plane_index, plane in enumerate(self.planes):
            image_2d_list = plane.image_2d_list_from(
                grid=traced_grid_list[plane_index], operated_only=operated_only
            )

            for galaxy_index, galaxy in enumerate(plane.galaxies):
                galaxy_image_2d_dict[galaxy] = image_2d_list[galaxy_index]

        return galaxy_image_2d_dict

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
    def upper_plane_index_with_light_profile(self) -> int:
        return max(
            [
                plane_index if plane.has(cls=ag.LightProfile) else 0
                for (plane_index, plane) in enumerate(self.planes)
            ]
        )

    @property
    def plane_indexes_with_pixelizations(self):
        plane_indexes_with_inversions = [
            plane_index if plane.has(cls=aa.Pixelization) else None
            for (plane_index, plane) in enumerate(self.planes)
        ]
        return [
            plane_index
            for plane_index in plane_indexes_with_inversions
            if plane_index is not None
        ]

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies in the `model_obj` have a `Pixelization` or `LightProfileLinear`
        object, in which case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """
        return any(plane.perform_inversion for plane in self.planes)

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class in the tracer as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if a tracer has an image-plane with a galaxy with two light profiles, the following:

        `tracer.extract_attribute(cls=LightProfile, name="axis_ratio")`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

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
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)

    def extract_attributes_of_planes(self, cls, attr_name, filter_nones=False):
        """
        Returns an attribute of a class in the tracer as a list of `ValueIrregular` or `Grid2DIrregular` objects, where
        the indexes of the list correspond to the tracer's planes.

        For example, if a tracer has an image-plane with a galaxy with a light profile and a source-plane with a galaxy
        with a light profile, the following:

        `tracer.extract_attributes_of_planes(cls=LightProfile, name="axis_ratio")`

        would return:

        [ArrayIrregular(values=[axis_ratio_0]), ArrayIrregular(values=[axis_ratio_1])]

        If the image plane has two galaxies with a mass profile each and the source plane another galaxy with a
        mass profile, the following:

        `tracer.extract_attributes_of_planes(cls=MassProfile, name="centres")`

        would return:

        [
            Grid2DIrregular(values=[(centre_y_0, centre_x_0)]),
            Grid2DIrregular(values=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)])
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

        [ArrayIrregular(values=[axis_ratio_0]), ArrayIrregular(values=[axis_ratio_1])]

        If the image plane has two galaxies with a mass profile each and the source plane another galaxy with a
        mass profile, the following:

        `tracer.extract_attributes_of_galaxies(cls=MassProfile, name="centres")`

        would return:

        [
            Grid2DIrregular(values=[(centre_y_0, centre_x_0)]),
            Grid2DIrregular(values=[(centre_y_0, centre_x_0)])
            Grid2DIrregular(values=[(centre_y_0, centre_x_0)])
        ]

        If the first galaxy in the image plane in the example above had two mass profiles as well as the galaxy in the
        source plane it would return:

                [
            Grid2DIrregular(values=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)]),
            Grid2DIrregular(values=[(centre_y_0, centre_x_0)])
            Grid2DIrregular(values=[(centre_y_0, centre_x_0, (centre_y_1, centre_x_1))])
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
        psf: Optional[aa.Kernel2D] = None,
    ):
        """
        Iterate over every `LightProfileSNR` in the tracer and set their `intensity` values to values which give
        their input `signal_to_noise_ratio` value, which is performed as follows:

        - Evaluate the image of each light profile on the input grid.
        - Blur this image with a PSF, if included.
        - Take the value of the brightest pixel.
        - Use an input `exposure_time` and `background_sky` (e.g. from the `SimulatorImaging` object) to determine
          what value of `intensity` gives the desired signal to noise ratio for the image.

        The intensity is set using an input grid, meaning that for strong lensing calculations the ray-traced grid
        can be used such that the S/N accounts for the magnification of a source galaxy.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        psf
            The psf of the simulated imaging which can change the S/N of the light profile due to spreading out
            the emission.
        """
        grid = aa.Grid2D.uniform(
            shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, sub_size=1
        )

        traced_grids_of_planes_list = self.traced_grid_2d_list_from(grid=grid)

        for plane_index, plane in enumerate(self.planes):
            for galaxy in plane.galaxies:
                for light_profile in galaxy.cls_list_from(cls=ag.LightProfile):
                    if isinstance(light_profile, LightProfileSNR):
                        light_profile.set_intensity_from(
                            grid=traced_grids_of_planes_list[plane_index],
                            exposure_time=exposure_time,
                            background_sky_level=background_sky_level,
                            psf=psf,
                        )

    @aa.profile_func
    def convolve_via_convolver(self, image, blurring_image, convolver):
        return convolver.convolve_image(image=image, blurring_image=blurring_image)
