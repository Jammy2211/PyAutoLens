from abc import ABC

import numpy as np
from astropy import cosmology as cosmo

from autoastro import lensing
from autoarray.util import array_util, grid_util
from autoarray.mask import mask as msk
from autoarray.structures import grids
from autoarray.structures.arrays import MaskedArray
from autoarray.operators.inversion import inversions as inv
from autoastro.galaxy import galaxy as g
from autoastro.util import cosmology_util
from autolens.lens import plane as pl
from autolens.util import lens_util


class AbstractTracer(lensing.LensingObject, ABC):
    def __init__(self, planes, cosmology):
        """Ray-tracer for a lens system with any number of planes.

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

    @property
    def total_planes(self):
        return len(self.plane_redshifts)

    @property
    def image_plane(self):
        return self.planes[0]

    @property
    def source_plane(self):
        return self.planes[-1]

    @property
    def galaxies(self):
        return list([galaxy for plane in self.planes for galaxy in plane.galaxies])

    @property
    def all_planes_have_redshifts(self):
        return None not in self.plane_redshifts

    @property
    def has_light_profile(self):
        return any(list(map(lambda plane: plane.has_light_profile, self.planes)))

    @property
    def has_mass_profile(self):
        return any(list(map(lambda plane: plane.has_mass_profile, self.planes)))

    @property
    def has_pixelization(self):
        return any(list(map(lambda plane: plane.has_pixelization, self.planes)))

    @property
    def has_regularization(self):
        return any(list(map(lambda plane: plane.has_regularization, self.planes)))

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda plane: plane.has_hyper_galaxy, self.planes)))

    @property
    def upper_plane_index_with_light_profile(self):
        return max(
            [
                plane_index if plane.has_light_profile else 0
                for (plane_index, plane) in enumerate(self.planes)
            ]
        )

    @property
    def planes_with_light_profile(self):
        return list(filter(lambda plane: plane.has_light_profile, self.planes))

    @property
    def planes_with_mass_profile(self):
        return list(filter(lambda plane: plane.has_mass_profile, self.planes))

    @property
    def light_profile_centres(self):
        return [
            item
            for light_profile_centres in self.light_profile_centres_of_planes
            for item in light_profile_centres
        ]

    @property
    def light_profile_centres_of_planes(self):
        return [
            plane.light_profile_centres
            for plane in self.planes
            if plane.has_light_profile
        ]

    @property
    def mass_profiles(self):
        return [
            item
            for mass_profile in self.mass_profiles_of_planes
            for item in mass_profile
        ]

    @property
    def mass_profiles_of_planes(self):
        return [plane.mass_profiles for plane in self.planes if plane.has_mass_profile]

    @property
    def mass_profile_centres(self):
        return [
            item
            for mass_profile_centres in self.mass_profile_centres_of_planes
            for item in mass_profile_centres
        ]

    @property
    def mass_profile_centres_of_planes(self):
        return [
            plane.mass_profile_centres
            for plane in self.planes
            if plane.has_mass_profile
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
    def pixelizations_of_planes(self):
        return [plane.pixelization for plane in self.planes]

    @property
    def regularizations_of_planes(self):
        return [plane.regularization for plane in self.planes]

    @property
    def hyper_galaxy_image_of_planes_with_pixelizations(self):
        return [
            plane.hyper_galaxy_image_of_galaxy_with_pixelization
            for plane in self.planes
        ]

    def plane_with_galaxy(self, galaxy):
        return [plane for plane in self.planes if galaxy in plane.galaxies][0]

    def new_object_with_units_converted(
        self,
        unit_length=None,
        unit_luminosity=None,
        unit_mass=None,
        kpc_per_arcsec=None,
        exposure_time=None,
        critical_surface_density=None,
    ):

        new_planes = list(
            map(
                lambda plane: plane.new_object_with_units_converted(
                    unit_length=unit_length,
                    unit_luminosity=unit_luminosity,
                    unit_mass=unit_mass,
                    kpc_per_arcsec=kpc_per_arcsec,
                    exposure_time=exposure_time,
                    critical_surface_density=critical_surface_density,
                ),
                self.planes,
            )
        )

        return self.__class__(planes=new_planes, cosmology=self.cosmology)

    @property
    def unit_length(self):
        if self.has_light_profile:
            return self.planes_with_light_profile[0].unit_length
        elif self.has_mass_profile:
            return self.planes_with_mass_profile[0].unit_length
        else:
            return None

    @property
    def unit_luminosity(self):
        if self.has_light_profile:
            return self.planes_with_light_profile[0].unit_luminosity
        elif self.has_mass_profile:
            return self.planes_with_mass_profile[0].unit_luminosity
        else:
            return None

    @property
    def unit_mass(self):
        if self.has_mass_profile:
            return self.planes_with_mass_profile[0].unit_mass
        else:
            return None


class AbstractTracerCosmology(AbstractTracer, ABC):
    def arcsec_per_kpc_proper_of_plane(self, i):
        return cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(
            redshift=self.plane_redshifts[i], cosmology=self.cosmology
        )

    def kpc_per_arcsec_proper_of_plane(self, i):
        return 1.0 / self.arcsec_per_kpc_proper_of_plane(i=i)

    def angular_diameter_distance_of_plane_to_earth_in_units(
        self, i, unit_length="arcsec"
    ):
        return cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
            redshift=self.plane_redshifts[i],
            cosmology=self.cosmology,
            unit_length=unit_length,
        )

    def angular_diameter_distance_between_planes_in_units(
        self, i, j, unit_length="arcsec"
    ):
        return cosmology_util.angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(
            redshift_0=self.plane_redshifts[i],
            redshift_1=self.plane_redshifts[j],
            cosmology=self.cosmology,
            unit_length=unit_length,
        )

    def angular_diameter_distance_to_source_plane_in_units(self, unit_length="arcsec"):
        return cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
            redshift=self.plane_redshifts[-1],
            cosmology=self.cosmology,
            unit_length=unit_length,
        )

    def critical_surface_density_between_planes_in_units(
        self, i, j, unit_length="arcsec", unit_mass="solMass"
    ):
        return cosmology_util.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
            redshift_0=self.plane_redshifts[i],
            redshift_1=self.plane_redshifts[j],
            cosmology=self.cosmology,
            unit_length=unit_length,
            unit_mass=unit_mass,
        )

    def scaling_factor_between_planes(self, i, j):
        return cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
            redshift_0=self.plane_redshifts[i],
            redshift_1=self.plane_redshifts[j],
            redshift_final=self.plane_redshifts[-1],
            cosmology=self.cosmology,
        )

    def angular_diameter_distance_from_image_to_source_plane_in_units(
        self, unit_length="arcsec"
    ):
        return self.angular_diameter_distance_between_planes_in_units(
            i=0, j=-1, unit_length=unit_length
        )


class AbstractTracerLensing(AbstractTracerCosmology, ABC):
    @grids.convert_coordinates_to_grid
    def traced_grids_of_planes_from_grid(self, grid, plane_index_limit=None):

        grid_calc = grid.copy()  # TODO looks unnecessary? Probably pretty expensive too

        traced_grids = []
        traced_deflections = []

        for (plane_index, plane) in enumerate(self.planes):

            scaled_grid = grid_calc.copy()

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=self.plane_redshifts[previous_plane_index],
                        redshift_1=plane.redshift,
                        redshift_final=self.plane_redshifts[-1],
                        cosmology=self.cosmology,
                    )

                    scaled_deflections = (
                        scaling_factor * traced_deflections[previous_plane_index]
                    )

                    scaled_grid -= scaled_deflections

            traced_grids.append(scaled_grid)

            if plane_index_limit is not None:
                if plane_index == plane_index_limit:
                    return traced_grids

            traced_deflections.append(plane.deflections_from_grid(grid=scaled_grid))

        return traced_grids

    @grids.convert_coordinates_to_grid
    def deflections_between_planes_from_grid(self, grid, plane_i=0, plane_j=-1):

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        return traced_grids_of_planes[plane_i] - traced_grids_of_planes[plane_j]

    @grids.convert_coordinates_to_grid
    def profile_image_from_grid(self, grid):
        profile_image = sum(self.profile_images_of_planes_from_grid(grid=grid))
        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=profile_image
        )

    @grids.convert_coordinates_to_grid
    def profile_images_of_planes_from_grid(self, grid):
        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, plane_index_limit=self.upper_plane_index_with_light_profile
        )

        profile_images_of_planes = [
            self.planes[plane_index].profile_image_from_grid(
                grid=traced_grids_of_planes[plane_index]
            )
            for plane_index in range(len(traced_grids_of_planes))
        ]

        if self.upper_plane_index_with_light_profile < self.total_planes - 1:
            for plane_index in range(
                self.upper_plane_index_with_light_profile, self.total_planes - 1
            ):
                profile_images_of_planes.append(
                    grid.mapping.array_stored_1d_from_sub_array_1d(
                        sub_array_1d=np.zeros(shape=profile_images_of_planes[0].shape)
                    )
                )

        return profile_images_of_planes

    def padded_profile_image_from_grid_and_psf_shape(self, grid, psf_shape_2d):

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf_shape_2d)

        return self.profile_image_from_grid(grid=padded_grid)

    @grids.convert_coordinates_to_grid
    def convergence_from_grid(self, grid):
        convergence = sum(
            [plane.convergence_from_grid(grid=grid) for plane in self.planes]
        )
        return grid.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=convergence)

    @grids.convert_coordinates_to_grid
    def potential_from_grid(self, grid):
        potential = sum([plane.potential_from_grid(grid=grid) for plane in self.planes])
        return grid.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=potential)

    @grids.convert_coordinates_to_grid
    def deflections_from_grid(self, grid):
        return self.deflections_between_planes_from_grid(grid=grid)

    @grids.convert_coordinates_to_grid
    def deflections_of_planes_summed_from_grid(self, grid):
        deflections = sum(
            [plane.deflections_from_grid(grid=grid) for plane in self.planes]
        )
        return grid.mapping.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=deflections)

    def grid_at_redshift_from_grid_and_redshift(self, grid, redshift):
        """For an input grid of (y,x) arc-second image-plane coordinates, ray-trace the coordinates to any redshift in \
        the strong lens configuration.

        This is performed using multi-plane ray-tracing and the existing redshifts and planes of the tracer. However, \
        any redshift can be input even if a plane does not exist there, including redshifts before the first plane \
        of the lens system.

        Parameters
        ----------
        grid : ndsrray or aa.Grid
            The image-plane grid which is traced to the redshift.
        redshift : float
            The redshift the image-plane grid is traced to.
        """

        if redshift <= self.plane_redshifts[0]:
            return grid.copy()

        plane_index_with_redshift = [
            plane_index
            for plane_index, plane in enumerate(self.planes)
            if plane.redshift == redshift
        ]

        if plane_index_with_redshift:
            return self.traced_grids_of_planes_from_grid(grid=grid)[
                plane_index_with_redshift[0]
            ]

        for plane_index, plane_redshift in enumerate(self.plane_redshifts):

            if redshift < plane_redshift:
                plane_index_insert = plane_index

        planes = self.planes
        planes.insert(
            plane_index_insert,
            pl.Plane(redshift=redshift, galaxies=[], cosmology=self.cosmology),
        )

        tracer = Tracer(planes=planes, cosmology=self.cosmology)

        return tracer.traced_grids_of_planes_from_grid(grid=grid)[plane_index_insert]

    def image_plane_multiple_image_positions_of_galaxies(self, grid):
        return [
            self.image_plane_multiple_image_positions(
                grid=grid, source_plane_coordinate=light_profile_centre
            )
            for light_profile_centre in self.light_profile_centres_of_planes[-1]
        ]

    def image_plane_multiple_image_positions(self, grid, source_plane_coordinate):

        # TODO : This should not input as a grid but use a iterative adaptive grid.

        if grid.sub_size > 1:
            grid = grid.in_1d_binned

        source_plane_grid = self.traced_grids_of_planes_from_grid(grid=grid)[-1]

        source_plane_squared_distances = source_plane_grid.squared_distances_from_coordinate(
            coordinate=source_plane_coordinate
        )

        trough_pixels = array_util.trough_pixels_from_array_2d(
            array_2d=source_plane_squared_distances.in_2d, mask_2d=grid.mask
        )

        trough_mask = msk.Mask.from_pixel_coordinates(
            shape_2d=grid.shape_2d,
            pixel_coordinates=trough_pixels,
            pixel_scales=grid.pixel_scales,
            sub_size=grid.sub_size,
            origin=grid.origin,
            buffer=1,
        )

        multiple_image_pixels = grid_util.positions_at_coordinate_from_grid_2d(
            grid_2d=source_plane_grid.in_2d,
            coordinate=source_plane_coordinate,
            mask_2d=trough_mask,
        )

        return grids.Coordinates.from_pixels_and_mask(
            pixels=[multiple_image_pixels], mask=trough_mask
        )

    @property
    def contribution_map(self):

        contribution_maps = self.contribution_maps_of_planes
        if None in contribution_maps:
            contribution_maps = [i for i in contribution_maps if i is not None]

        if contribution_maps:
            return sum(contribution_maps)
        else:
            return None

    @property
    def contribution_maps_of_planes(self):

        contribution_maps = []

        for plane in self.planes:

            if plane.contribution_map is not None:

                contribution_maps.append(plane.contribution_map)

            else:

                contribution_maps.append(None)

        return contribution_maps


class AbstractTracerData(AbstractTracerLensing, ABC):
    def blurred_profile_image_from_grid_and_psf(self, grid, psf, blurring_grid):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a psf (see imaging.convolution).

        These are summed to give the tracer's overall blurred image in 1D.

        Parameters
        ----------
        psf : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        profile_image = self.profile_image_from_grid(grid=grid)

        blurring_image = self.profile_image_from_grid(grid=blurring_grid)

        return psf.convolved_array_from_array_2d_and_mask(
            array_2d=profile_image.in_2d_binned + blurring_image.in_2d_binned,
            mask=grid.mask,
        )

    def blurred_profile_images_of_planes_from_grid_and_psf(
        self, grid, psf, blurring_grid
    ):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a psf (see imaging.convolution).

        The blurred image of every plane is returned in 1D.

        Parameters
        ----------
        psf : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)
        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid
        )

        return [
            plane.blurred_profile_image_from_grid_and_psf(
                grid=traced_grids_of_planes[plane_index],
                psf=psf,
                blurring_grid=traced_blurring_grids_of_planes[plane_index],
            )
            for (plane_index, plane) in enumerate(self.planes)
        ]

    def blurred_profile_image_from_grid_and_convolver(
        self, grid, convolver, blurring_grid
    ):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a convolver (see imaging.convolution).

        These are summed to give the tracer's overall blurred image in 1D.

        Parameters
        ----------
        convolver : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        profile_image = self.profile_image_from_grid(grid=grid)

        blurring_image = self.profile_image_from_grid(grid=blurring_grid)

        return convolver.convolved_image_from_image_and_blurring_image(
            image=profile_image, blurring_image=blurring_image
        )

    def blurred_profile_images_of_planes_from_grid_and_convolver(
        self, grid, convolver, blurring_grid
    ):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a convolver (see imaging.convolution).

        The blurred image of every plane is returned in 1D.

        Parameters
        ----------
        convolver : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)
        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid
        )

        return [
            plane.blurred_profile_image_from_grid_and_convolver(
                grid=traced_grids_of_planes[plane_index],
                convolver=convolver,
                blurring_grid=traced_blurring_grids_of_planes[plane_index],
            )
            for (plane_index, plane) in enumerate(self.planes)
        ]

    def unmasked_blurred_profile_image_from_grid_and_psf(self, grid, psf):

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf.shape_2d)

        padded_image = self.profile_image_from_grid(grid=padded_grid)

        return padded_grid.mapping.unmasked_blurred_array_from_padded_array_psf_and_image_shape(
            padded_array=padded_image, psf=psf, image_shape=grid.mask.shape
        )

    def unmasked_blurred_profile_image_of_planes_from_grid_and_psf(self, grid, psf):

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf.shape_2d)

        traced_padded_grids = self.traced_grids_of_planes_from_grid(grid=padded_grid)

        unmasked_blurred_profile_images_of_planes = []

        for plane, traced_padded_grid in zip(self.planes, traced_padded_grids):
            padded_image_1d = plane.profile_image_from_grid(grid=traced_padded_grid)

            unmasked_blurred_array_2d = padded_grid.mapping.unmasked_blurred_array_from_padded_array_psf_and_image_shape(
                padded_array=padded_image_1d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_profile_images_of_planes.append(unmasked_blurred_array_2d)

        return unmasked_blurred_profile_images_of_planes

    def unmasked_blurred_profile_image_of_planes_and_galaxies_from_grid_and_psf(
        self, grid, psf
    ):

        unmasked_blurred_profile_images_of_planes_and_galaxies = []

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf.shape_2d)

        traced_padded_grids = self.traced_grids_of_planes_from_grid(grid=padded_grid)

        for plane, traced_padded_grid in zip(self.planes, traced_padded_grids):
            padded_image_1d_of_galaxies = plane.profile_images_of_galaxies_from_grid(
                grid=traced_padded_grid
            )

            unmasked_blurred_array_2d_of_galaxies = list(
                map(
                    lambda padded_image_1d_of_galaxy: padded_grid.mapping.unmasked_blurred_array_from_padded_array_psf_and_image_shape(
                        padded_array=padded_image_1d_of_galaxy,
                        psf=psf,
                        image_shape=grid.mask.shape,
                    ),
                    padded_image_1d_of_galaxies,
                )
            )

            unmasked_blurred_profile_images_of_planes_and_galaxies.append(
                unmasked_blurred_array_2d_of_galaxies
            )

        return unmasked_blurred_profile_images_of_planes_and_galaxies

    def profile_visibilities_from_grid_and_transformer(self, grid, transformer):

        profile_image = self.profile_image_from_grid(grid=grid)
        return transformer.visibilities_from_image(image=profile_image)

    def profile_visibilities_of_planes_from_grid_and_transformer(
        self, grid, transformer
    ):

        profile_images_of_planes = self.profile_images_of_planes_from_grid(grid=grid)
        return [
            transformer.visibilities_from_image(image=profile_image)
            for profile_image in profile_images_of_planes
        ]

    def sparse_image_plane_grids_of_planes_from_grid(self, grid):

        sparse_image_plane_grids_of_planes = []

        for plane in self.planes:
            sparse_image_plane_grid = plane.sparse_image_plane_grid_from_grid(grid=grid)
            sparse_image_plane_grids_of_planes.append(sparse_image_plane_grid)

        return sparse_image_plane_grids_of_planes

    def traced_sparse_grids_of_planes_from_grid(
        self, grid, preload_sparse_grids_of_planes=None
    ):

        if preload_sparse_grids_of_planes is None:

            sparse_image_plane_grids_of_planes = self.sparse_image_plane_grids_of_planes_from_grid(
                grid=grid
            )

        else:

            sparse_image_plane_grids_of_planes = preload_sparse_grids_of_planes

        traced_sparse_grids_of_planes = []

        for (plane_index, plane) in enumerate(self.planes):

            if sparse_image_plane_grids_of_planes[plane_index] is None:
                traced_sparse_grids_of_planes.append(None)
            else:
                traced_sparse_grids = self.traced_grids_of_planes_from_grid(
                    grid=sparse_image_plane_grids_of_planes[plane_index]
                )
                traced_sparse_grids_of_planes.append(traced_sparse_grids[plane_index])

        return traced_sparse_grids_of_planes

    def mappers_of_planes_from_grid(
        self, grid, inversion_uses_border=False, preload_sparse_grids_of_planes=None
    ):

        mappers_of_planes = []

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        traced_sparse_grids_of_planes = self.traced_sparse_grids_of_planes_from_grid(
            grid=grid, preload_sparse_grids_of_planes=preload_sparse_grids_of_planes
        )

        for (plane_index, plane) in enumerate(self.planes):

            if not plane.has_pixelization:
                mappers_of_planes.append(None)
            else:
                mapper = plane.mapper_from_grid_and_sparse_grid(
                    grid=traced_grids_of_planes[plane_index],
                    sparse_grid=traced_sparse_grids_of_planes[plane_index],
                    inversion_uses_border=inversion_uses_border,
                )
                mappers_of_planes.append(mapper)

        return mappers_of_planes

    def inversion_imaging_from_grid_and_data(
        self,
        grid,
        image,
        noise_map,
        convolver,
        inversion_uses_border=False,
        preload_sparse_grids_of_planes=None,
    ):

        mappers_of_planes = self.mappers_of_planes_from_grid(
            grid=grid,
            inversion_uses_border=inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return inv.InversionImaging.from_data_mapper_and_regularization(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            mapper=mappers_of_planes[-1],
            regularization=self.regularizations_of_planes[-1],
        )

    def inversion_interferometer_from_grid_and_data(
        self,
        grid,
        visibilities,
        noise_map,
        transformer,
        inversion_uses_border=False,
        preload_sparse_grids_of_planes=None,
    ):
        mappers_of_planes = self.mappers_of_planes_from_grid(
            grid=grid,
            inversion_uses_border=inversion_uses_border,
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes,
        )

        return inv.InversionInterferometer.from_data_mapper_and_regularization(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mappers_of_planes[-1],
            regularization=self.regularizations_of_planes[-1],
        )

    def hyper_noise_map_from_noise_map(self, noise_map):
        hyper_noise_maps = self.hyper_noise_maps_of_planes_from_noise_map(
            noise_map=noise_map
        )
        return sum(hyper_noise_maps)

    def hyper_noise_maps_of_planes_from_noise_map(self, noise_map):
        return [
            plane.hyper_noise_map_from_noise_map(noise_map=noise_map)
            for plane in self.planes
        ]

    def galaxy_profile_image_dict_from_grid(self, grid) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_profile_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        for (plane_index, plane) in enumerate(self.planes):
            profile_images_of_galaxies = plane.profile_images_of_galaxies_from_grid(
                grid=traced_grids_of_planes[plane_index]
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_profile_image_dict[galaxy] = profile_images_of_galaxies[
                    galaxy_index
                ]

        return galaxy_profile_image_dict

    def galaxy_blurred_profile_image_dict_from_grid_and_convolver(
        self, grid, convolver, blurring_grid
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_blurred_profile_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid
        )

        for (plane_index, plane) in enumerate(self.planes):
            blurred_profile_images_of_galaxies = plane.blurred_profile_images_of_galaxies_from_grid_and_convolver(
                grid=traced_grids_of_planes[plane_index],
                convolver=convolver,
                blurring_grid=traced_blurring_grids_of_planes[plane_index],
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_blurred_profile_image_dict[
                    galaxy
                ] = blurred_profile_images_of_galaxies[galaxy_index]

        return galaxy_blurred_profile_image_dict

    def galaxy_profile_visibilities_dict_from_grid_and_transformer(
        self, grid, transformer
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_profile_visibilities_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        for (plane_index, plane) in enumerate(self.planes):
            profile_visibilities_of_galaxies = plane.profile_visibilities_of_galaxies_from_grid_and_transformer(
                grid=traced_grids_of_planes[plane_index], transformer=transformer
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_profile_visibilities_image_dict[
                    galaxy
                ] = profile_visibilities_of_galaxies[galaxy_index]

        return galaxy_profile_visibilities_image_dict


class Tracer(AbstractTracerData):
    @classmethod
    def from_galaxies(cls, galaxies, cosmology=cosmo.Planck15):

        plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(
            galaxies=galaxies
        )

        galaxies_in_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=plane_redshifts
        )

        planes = []

        for plane_index in range(0, len(plane_redshifts)):
            planes.append(
                pl.Plane(galaxies=galaxies_in_planes[plane_index], cosmology=cosmology)
            )

        return Tracer(planes=planes, cosmology=cosmology)

    @classmethod
    def sliced_tracer_from_lens_line_of_sight_and_source_galaxies(
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

        lens_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(
            galaxies=lens_galaxies
        )

        plane_redshifts = lens_util.ordered_plane_redshifts_from_lens_source_plane_redshifts_and_slice_sizes(
            lens_redshifts=lens_redshifts,
            planes_between_lenses=planes_between_lenses,
            source_plane_redshift=source_galaxies[0].redshift,
        )

        galaxies_in_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=lens_galaxies + line_of_sight_galaxies,
            plane_redshifts=plane_redshifts,
        )

        plane_redshifts.append(source_galaxies[0].redshift)
        galaxies_in_planes.append(source_galaxies)

        planes = []

        for plane_index in range(0, len(plane_redshifts)):
            planes.append(
                pl.Plane(
                    redshift=plane_redshifts[plane_index],
                    galaxies=galaxies_in_planes[plane_index],
                    cosmology=cosmology,
                )
            )

        return Tracer(planes=planes, cosmology=cosmology)
