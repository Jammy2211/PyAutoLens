import numpy as np
from astropy import cosmology as cosmo

from autolens import exc
from autolens.array import grids
from autolens.lens import plane as pl
from autolens.lens.util import lens_util
from autolens.model import cosmology_util
from autolens.model.inversion import inversions as inv
from autolens.model.galaxy import galaxy as g

from autolens.array.mapping import (
    reshape_returned_sub_array,
    reshape_returned_array,
    reshape_returned_array,
    reshape_returned_grid,
)


class AbstractTracer(object):
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
            The image-plane al.ogrid which is traced. (includes the grid, sub-grid, blurring-grid, etc.).
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
    def hyper_galaxy_image_1d_of_planes_with_pixelizations(self):
        return [
            plane.binned_hyper_galaxy_image_1d_of_galaxy_with_pixelization
            for plane in self.planes
        ]

    def plane_with_galaxy(self, galaxy):
        return [plane for plane in self.planes if galaxy in plane.galaxies][0]


class AbstractTracerCosmology(AbstractTracer):
    def __init__(self, planes, cosmology):

        super(AbstractTracerCosmology, self).__init__(
            planes=planes, cosmology=cosmology
        )

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


class AbstractTracerLensing(AbstractTracerCosmology):
    def __init__(self, planes, cosmology):
        super(AbstractTracerLensing, self).__init__(planes=planes, cosmology=cosmology)

    def traced_grids_of_planes_from_grid(
        self, grid, return_in_2d=True, plane_index_limit=None
    ):

        grid_calc = grid.copy()

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

            traced_deflections.append(
                plane.deflections_from_grid(
                    grid=scaled_grid,
                    return_in_2d=False,
                    return_binned=False,
                    bypass_decorator=True,
                )
            )

        return traced_grids

    def traced_positions_of_planes_from_positions(self, positions):

        traced_positions_of_planes = [[] for i in range(self.total_planes)]

        for position_grid in positions:
            traced_position_grids_of_planes = self.traced_grids_of_planes_from_grid(
                grid=position_grid
            )
            for (plane_index, tracer_position_grid_of_plane) in enumerate(
                traced_position_grids_of_planes
            ):
                traced_positions_of_planes[plane_index].append(
                    traced_position_grids_of_planes[plane_index]
                )

        return traced_positions_of_planes

    def deflections_between_planes_from_grid(
        self, grid, plane_i=0, plane_j=-1, return_in_2d=True
    ):

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, return_in_2d=return_in_2d
        )

        return traced_grids_of_planes[plane_i] - traced_grids_of_planes[plane_j]

    @reshape_returned_sub_array
    def profile_image_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        return sum(
            self.profile_images_of_planes_from_grid(
                grid=grid,
                return_in_2d=False,
                return_binned=False,
                bypass_decorator=True,
            )
        )

    def profile_images_of_planes_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, plane_index_limit=self.upper_plane_index_with_light_profile
        )

        profile_images_of_planes = [
            self.planes[plane_index].profile_image_from_grid(
                grid=traced_grids_of_planes[plane_index],
                return_in_2d=return_in_2d,
                return_binned=return_binned,
                bypass_decorator=bypass_decorator,
            )
            for plane_index in range(len(traced_grids_of_planes))
        ]

        if self.upper_plane_index_with_light_profile < self.total_planes - 1:
            for plane_index in range(
                self.upper_plane_index_with_light_profile, self.total_planes - 1
            ):
                profile_images_of_planes.append(
                    np.zeros(shape=profile_images_of_planes[0].shape)
                )

        return profile_images_of_planes

    def padded_profile_image_2d_from_grid_and_psf_shape(self, grid, psf_shape):

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=psf_shape)

        return self.profile_image_from_grid(
            grid=padded_grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )

    @reshape_returned_sub_array
    def convergence_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        return sum(
            [
                plane.convergence_from_grid(
                    grid=grid,
                    return_in_2d=False,
                    return_binned=False,
                    bypass_decorator=True,
                )
                for plane in self.planes
            ]
        )

    @reshape_returned_sub_array
    def potential_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        return sum(
            [
                plane.potential_from_grid(
                    grid=grid,
                    return_in_2d=False,
                    return_binned=False,
                    bypass_decorator=True,
                )
                for plane in self.planes
            ]
        )

    @reshape_returned_grid
    def deflections_from_grid(
        self, grid, return_in_2d=True, return_binned=True, bypass_decorator=False
    ):
        return sum(
            [
                plane.deflections_from_grid(
                    grid=grid,
                    return_in_2d=False,
                    return_binned=False,
                    bypass_decorator=True,
                )
                for plane in self.planes
            ]
        )

    def einstein_radius_of_plane_in_units(self, i, unit_length="arcsec"):
        return self.planes[i].einstein_radius_in_units(unit_length=unit_length)

    def einstein_mass_between_planes_in_units(self, i, j, unit_mass="solMass"):
        return self.planes[i].einstein_mass_in_units(
            unit_mass=unit_mass, redshift_source=self.plane_redshifts[j]
        )

    def grid_at_redshift_from_grid_and_redshift(self, grid, redshift):
        """For an input grid of (y,x) arc-second image-plane coordinates, ray-trace the coordinates to any redshift in \
        the strong lens configuration.

        This is performed using multi-plane ray-tracing and the existing redshifts and planes of the tracer. However, \
        any redshift can be input even if a plane does not exist there, including redshifts before the first plane \
        of the lensing system.

        Parameters
        ----------
        grid : ndsrray or grids.Grid
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


class AbstractTracerData(AbstractTracerLensing):
    def __init__(self, planes, cosmology):
        super(AbstractTracerData, self).__init__(planes=planes, cosmology=cosmology)

    @reshape_returned_array
    def blurred_profile_image_from_grid_and_psf(
        self, grid, psf, blurring_grid, return_in_2d=True, bypass_decorator=False
    ):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a psf (see imaging.convolution).

        These are summed to give the tracer's overall blurred image in 1D.

        Parameters
        ----------
        psf : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        profile_image = self.profile_image_from_grid(
            grid=grid, return_in_2d=True, return_binned=True, bypass_decorator=False
        )

        blurring_image = self.profile_image_from_grid(
            grid=blurring_grid,
            return_in_2d=True,
            return_binned=True,
            bypass_decorator=False,
        )

        return psf.convolve(profile_image + blurring_image)

    def blurred_profile_images_of_planes_from_grid_and_psf(
        self, grid, psf, blurring_grid, return_in_2d=False
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
                return_in_2d=return_in_2d,
            )
            for (plane_index, plane) in enumerate(self.planes)
        ]

    @reshape_returned_array
    def blurred_profile_image_from_grid_and_convolver(
        self, grid, convolver, blurring_grid, return_in_2d=True, bypass_decorator=False
    ):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a convolver (see imaging.convolution).

        These are summed to give the tracer's overall blurred image in 1D.

        Parameters
        ----------
        convolver : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        image_array = self.profile_image_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )

        if convolver.blurring_mask is None:
            blurring_mask = grid.mask.blurring_mask_from_psf_shape(
                psf_shape=convolver.psf.shape
            )
            convolver = convolver.convolver_with_blurring_mask_added(
                blurring_mask=blurring_mask
            )

        blurring_array = self.profile_image_from_grid(
            grid=blurring_grid, return_in_2d=False, return_binned=True
        )

        return convolver.convolve_image(
            image_array=image_array, blurring_array=blurring_array
        )

    def blurred_profile_images_of_planes_from_grid_and_convolver(
        self, grid, convolver, blurring_grid, return_in_2d=False
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
                return_in_2d=return_in_2d,
            )
            for (plane_index, plane) in enumerate(self.planes)
        ]

    def unmasked_blurred_profile_image_from_grid_and_psf(self, grid, psf):

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=psf.shape)

        padded_image_1d = self.profile_image_from_grid(
            grid=padded_grid, return_in_2d=False, return_binned=True
        )

        return padded_grid.unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
            padded_array_1d=padded_image_1d, psf=psf, image_shape=grid.mask.shape
        )

    def unmasked_blurred_profile_image_of_planes_from_grid_and_psf(self, grid, psf):

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=psf.shape)

        traced_padded_grids = self.traced_grids_of_planes_from_grid(grid=padded_grid)

        unmasked_blurred_profile_images_of_planes = []

        for plane, traced_padded_grid in zip(self.planes, traced_padded_grids):

            padded_image_1d = plane.profile_image_from_grid(
                grid=traced_padded_grid, return_in_2d=False, return_binned=True
            )

            unmasked_blurred_array_2d = padded_grid.unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
                padded_array_1d=padded_image_1d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_profile_images_of_planes.append(unmasked_blurred_array_2d)

        return unmasked_blurred_profile_images_of_planes

    def unmasked_blurred_profile_image_of_planes_and_galaxies_from_grid_and_psf(
        self, grid, psf
    ):

        unmasked_blurred_profile_images_of_planes_and_galaxies = []

        padded_grid = grid.padded_grid_from_psf_shape(psf_shape=psf.shape)

        traced_padded_grids = self.traced_grids_of_planes_from_grid(grid=padded_grid)

        for plane, traced_padded_grid in zip(self.planes, traced_padded_grids):

            padded_image_1d_of_galaxies = plane.profile_images_of_galaxies_from_grid(
                grid=traced_padded_grid, return_in_2d=False, return_binned=True
            )

            unmasked_blurred_array_2d_of_galaxies = list(
                map(
                    lambda padded_image_1d_of_galaxy: padded_grid.unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
                        padded_array_1d=padded_image_1d_of_galaxy,
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

        profile_image_1d = self.profile_image_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )
        return transformer.visibilities_from_image_1d(image_1d=profile_image_1d)

    def profile_visibilities_of_planes_from_grid_and_transformer(
        self, grid, transformer
    ):

        profile_images_1d_of_planes = self.profile_images_of_planes_from_grid(
            grid=grid, return_in_2d=False, return_binned=True
        )
        return [
            transformer.visibilities_from_image_1d(image_1d=profile_image_1d)
            for profile_image_1d in profile_images_1d_of_planes
        ]

    def pixelization_grids_of_planes_from_grid(self, grid):

        pixelization_grids_of_planes = []

        for plane in self.planes:
            pixelization_grid = plane.pixelization_grid_from_grid(grid=grid)
            pixelization_grids_of_planes.append(pixelization_grid)

        return pixelization_grids_of_planes

    def traced_pixelization_grids_of_planes_from_grid(
        self, grid, preload_pixelization_grids_of_planes=None
    ):

        if preload_pixelization_grids_of_planes is None:

            pixelization_grids_of_planes = self.pixelization_grids_of_planes_from_grid(
                grid=grid
            )

        else:

            pixelization_grids_of_planes = preload_pixelization_grids_of_planes

        traced_pixelization_grids_of_planes = []

        for (plane_index, plane) in enumerate(self.planes):

            if pixelization_grids_of_planes[plane_index] is None:
                traced_pixelization_grids_of_planes.append(None)
            else:
                traced_pixelization_grids = self.traced_grids_of_planes_from_grid(
                    grid=pixelization_grids_of_planes[plane_index]
                )
                traced_pixelization_grids_of_planes.append(
                    traced_pixelization_grids[plane_index]
                )

        return traced_pixelization_grids_of_planes

    def mappers_of_planes_from_grid(
        self,
        grid,
        inversion_uses_border=False,
        preload_pixelization_grids_of_planes=None,
    ):

        mappers_of_planes = []

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        traced_pixelization_grids_of_planes = self.traced_pixelization_grids_of_planes_from_grid(
            grid=grid,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        for (plane_index, plane) in enumerate(self.planes):

            if not plane.has_pixelization:
                mappers_of_planes.append(None)
            else:
                mapper = plane.mapper_from_grid_and_pixelization_grid(
                    grid=traced_grids_of_planes[plane_index],
                    pixelization_grid=traced_pixelization_grids_of_planes[plane_index],
                    inversion_uses_border=inversion_uses_border,
                )
                mappers_of_planes.append(mapper)

        return mappers_of_planes

    def inversion_from_grid_image_1d_noise_map_1d_and_convolver(
        self,
        grid,
        image_1d,
        noise_map_1d,
        convolver,
        inversion_uses_border=False,
        preload_pixelization_grids_of_planes=None,
    ):

        mappers_of_planes = self.mappers_of_planes_from_grid(
            grid=grid,
            inversion_uses_border=inversion_uses_border,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        return inv.Inversion.from_data_1d_mapper_and_regularization(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            convolver=convolver,
            mapper=mappers_of_planes[-1],
            regularization=self.regularizations_of_planes[-1],
        )

    def hyper_noise_map_1d_from_noise_map_1d(self, noise_map_1d):
        hyper_noise_maps_1d = self.hyper_noise_maps_1d_of_planes_from_noise_map_1d(
            noise_map_1d=noise_map_1d
        )
        hyper_noise_maps_1d = [
            hyper_noise_map
            for hyper_noise_map in hyper_noise_maps_1d
            if hyper_noise_map is not None
        ]
        return sum(hyper_noise_maps_1d)

    def hyper_noise_maps_1d_of_planes_from_noise_map_1d(self, noise_map_1d):
        return [
            plane.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            for plane in self.planes
        ]

    def galaxy_profile_image_dict_from_grid(
        self, grid, return_in_2d=True, return_binned=True
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_profile_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, return_in_2d=False
        )

        for (plane_index, plane) in enumerate(self.planes):
            profile_images_of_galaxies = plane.profile_images_of_galaxies_from_grid(
                grid=traced_grids_of_planes[plane_index],
                return_in_2d=return_in_2d,
                return_binned=return_binned,
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_profile_image_dict[galaxy] = profile_images_of_galaxies[
                    galaxy_index
                ]

        return galaxy_profile_image_dict

    def galaxy_blurred_profile_image_dict_from_grid_and_convolver(
        self, grid, convolver, blurring_grid, return_in_2d=True
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_blurred_profile_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, return_in_2d=False
        )

        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid, return_in_2d=False
        )

        for (plane_index, plane) in enumerate(self.planes):
            blurred_profile_images_of_galaxies = plane.blurred_profile_images_of_galaxies_from_grid_and_convolver(
                grid=traced_grids_of_planes[plane_index],
                convolver=convolver,
                blurring_grid=traced_blurring_grids_of_planes[plane_index],
                return_in_2d=return_in_2d,
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

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, return_in_2d=False
        )

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
    def __init__(self, planes, cosmology):
        super(AbstractTracerData, self).__init__(planes=planes, cosmology=cosmology)

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
            The image-plane al.ogrid which is traced. (includes the grid, sub-grid, blurring-grid, etc.).
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
