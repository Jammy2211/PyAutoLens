import numpy as np
from astropy import cosmology as cosmo
from functools import wraps

from autolens import exc
from autolens.data.array import grids
from autolens.lens import plane as pl
from autolens.lens.util import lens_util
from autolens.model import cosmology_util
from autolens.model.inversion import inversions as inv
from autolens.model.galaxy import galaxy as g

from autolens.data.array.grids import (
    reshape_returned_array,
    reshape_returned_array_blurring,
    reshape_returned_grid,
)


class AbstractTracerCosmology(object):
    def __init__(self, plane_redshifts, cosmology):
        """Abstract Ray tracer for lens systems with any number of planes.

        From the galaxies of the tracer's planes, their grid-stack(s) and the cosmology physically derived quantities \
        (e.g. surface density, angular diameter distances, critical surface densities) can be computed.

        Parameters
        ----------
        plane_redshifts : [pl.Plane]
            The list of the tracer's planes in ascending redshift order.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """
        self.plane_redshifts = plane_redshifts
        self.cosmology = cosmology

    @property
    def total_planes(self):
        return len(self.plane_redshifts)

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

    def padded_tracer_from_psf_shape(self, psf_shape):
        raise NotImplementedError()


class AbstractTracer(AbstractTracerCosmology):
    def __init__(self, planes, cosmology):
        """Abstract Ray tracer for lens systems with any number of planes.

        From the galaxies of the tracer's planes, their grid-stack(s) and the cosmology physically derived quantities \
        (e.g. surface density, angular diameter distances, critical surface densities) can be computed.

        Parameters
        ----------
        planes : [pl.Plane]
            The list of the tracer's planes in ascending redshift order.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        self.planes = planes
        super(AbstractTracer, self).__init__(
            plane_redshifts=[plane.redshift for plane in self.planes],
            cosmology=cosmology,
        )

    def plane_with_galaxy(self, galaxy):
        return [plane for plane in self.planes if galaxy in plane.galaxies][0]

    @property
    def grid_stack(self):
        return self.planes[0].grid_stack

    @property
    def image_plane(self):
        return self.planes[0]

    @property
    def source_plane(self):
        return self.planes[-1]

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
    def galaxies(self):
        return list([galaxy for plane in self.planes for galaxy in plane.galaxies])

    @reshape_returned_array
    def profile_image_plane_image(self, return_in_2d=True, return_binned=True):
        return sum(
            self.profile_image_plane_image_of_planes(
                return_in_2d=False, return_binned=False
            )
        )

    def profile_image_plane_image_of_planes(
        self, return_in_2d=True, return_binned=True
    ):
        return [
            plane.profile_image_plane_image(
                return_in_2d=return_in_2d, return_binned=return_binned
            )
            for plane in self.planes
        ]

    def padded_profile_image_plane_image_2d_from_psf_shape(self, psf_shape):

        padded_tracer = self.padded_tracer_from_psf_shape(psf_shape=psf_shape)

        return padded_tracer.profile_image_plane_image(
            return_in_2d=True, return_binned=True
        )

    @reshape_returned_array_blurring
    def profile_image_plane_blurring_image(self, return_in_2d=True):
        return sum(
            self.profile_image_plane_blurring_image_of_planes(return_in_2d=False)
        )

    def profile_image_plane_blurring_image_of_planes(self, return_in_2d=True):
        return [
            plane.profile_image_plane_blurring_image(return_in_2d=return_in_2d)
            for plane in self.planes
        ]

    @property
    def mappers_of_planes(self):
        return list(filter(None, [plane.mapper for plane in self.planes]))

    @property
    def regularizations_of_planes(self):
        return list(filter(None, [plane.regularization for plane in self.planes]))

    @reshape_returned_array
    def convergence(self, return_in_2d=True, return_binned=True):
        return sum(
            [
                plane.convergence(return_in_2d=False, return_binned=False)
                for plane in self.planes
            ]
        )

    @reshape_returned_array
    def potential(self, return_in_2d=True, return_binned=True):
        return sum(
            [
                plane.potential(return_in_2d=False, return_binned=False)
                for plane in self.planes
            ]
        )

    @reshape_returned_array
    def deflections_y(self, return_in_2d=True, return_binned=True):
        return sum(
            [
                plane.deflections_y(return_in_2d=False, return_binned=False)
                for plane in self.planes
            ]
        )

    @reshape_returned_array
    def deflections_x(self, return_in_2d=True, return_binned=True):
        return sum(
            [
                plane.deflections_x(return_in_2d=False, return_binned=False)
                for plane in self.planes
            ]
        )

    @reshape_returned_grid
    def deflections(self, return_in_2d=True, return_binned=True):
        return sum(
            [
                plane.deflections(return_in_2d=False, return_binned=False)
                for plane in self.planes
            ]
        )

    def einstein_radius_of_plane_in_units(self, i, unit_length="arcsec"):
        return self.planes[i].einstein_radius_in_units(unit_length=unit_length)

    def einstein_mass_between_planes_in_units(self, i, j, unit_mass="solMass"):
        return self.planes[i].einstein_mass_in_units(
            unit_mass=unit_mass, redshift_source=self.plane_redshifts[j]
        )

    def grid_at_redshift_from_image_plane_grid_and_redshift(
        self, image_plane_grid, redshift
    ):
        """For an input grid of (y,x) arc-second image-plane coordinates, ray-trace the coordinates to any redshift in \
        the strong lens configuration.

        This is performed using multi-plane ray-tracing and the existing redshifts and planes of the tracer. However, \
        any redshift can be input even if a plane does not exist there, including redshifts before the first plane \
        of the lensing system.

        Parameters
        ----------
        image_plane_grid : ndsrray or grids.Grid
            The image-plane grid which is traced to the redshift.
        redshift : float
            The redshift the image-plane grid is traced to.
        """

        # TODO : We need to come up with a better abstraction for multi-plane lensing 0_0

        image_plane_grid_stack = grids.GridStack(
            regular=image_plane_grid,
            sub=np.array([[0.0, 0.0]]),
            blurring=np.array([[0.0, 0.0]]),
        )

        tracer = Tracer.from_galaxies_and_image_plane_grid_stack(
            galaxies=self.galaxies,
            image_plane_grid_stack=image_plane_grid_stack,
            border=None,
            cosmology=self.cosmology,
        )

        for plane_index in range(0, len(self.plane_redshifts)):

            new_grid_stack = image_plane_grid_stack

            if redshift <= tracer.plane_redshifts[plane_index]:

                # If redshift is between two planes, we need to map over all previous planes coordinates / deflections.

                if plane_index > 0:
                    for previous_plane_index in range(plane_index):
                        scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                            redshift_0=tracer.plane_redshifts[previous_plane_index],
                            redshift_1=redshift,
                            redshift_final=tracer.plane_redshifts[-1],
                            cosmology=tracer.cosmology,
                        )

                        scaled_deflections_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
                            plane=tracer.planes[previous_plane_index],
                            scaling_factor=scaling_factor,
                        )

                        new_grid_stack = lens_util.grid_stack_from_deflections_stack(
                            grid_stack=new_grid_stack,
                            deflections_stack=scaled_deflections_stack,
                        )

                # If redshift is before the first plane, no change to image pllane coordinates.

                elif plane_index == 0:

                    return new_grid_stack.regular

                return new_grid_stack.regular


class AbstractTracerData(AbstractTracer):
    def __init__(self, planes, cosmology):
        """Abstract Ray tracer for lens systems with any number of planes.

        From the galaxies of the tracer's planes, their grid-stack(s) and the cosmology physically derived quantities \
        (e.g. surface density, angular diameter distances, critical surface densities) can be computed.

        Parameters
        ----------
        planes : [pl.Plane]
            The list of the tracer's planes in ascending redshift order.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """
        super(AbstractTracerData, self).__init__(planes=planes, cosmology=cosmology)

    def blurred_profile_image_plane_image_2d_of_planes_from_convolver_image(
        self, convolver_image
    ):
        """Extract the 1D image-plane image and 1D blurring image-plane image of every plane and blur each with the \
        PSF using a convolver (see ccd.convolution) and then map them back to the 2D array of the original mask.

        The blurred image of every plane is returned in 2D.

        Parameters
        ----------
        convolver_image : hyper.ccd.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        blurred_profile_image_plane_images_of_planes_1d = self.blurred_profile_image_plane_image_1d_of_planes_from_convolver_image(
            convolver_image=convolver_image
        )

        return list(
            map(
                lambda blurred_profile_image_plane_image_1d: self.image_plane.grid_stack.scaled_array_2d_from_array_1d(
                    array_1d=blurred_profile_image_plane_image_1d
                ),
                blurred_profile_image_plane_images_of_planes_1d,
            )
        )

    def blurred_profile_image_plane_image_1d_from_convolver_image(
        self, convolver_image
    ):
        """Extract the 1D image-plane image and 1D blurring image-plane image of every plane and blur each with the \
        PSF using a convolver (see ccd.convolution).

        These are summed to give the tracer's overall blurred image-plane image in 1D.

        Parameters
        ----------
        convolver_image : hyper.ccd.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        image_array = self.profile_image_plane_image(
            return_in_2d=False, return_binned=True
        )
        blurring_array = self.profile_image_plane_blurring_image(return_in_2d=False)

        return convolver_image.convolve_image(
            image_array=image_array, blurring_array=blurring_array
        )

    def blurred_profile_image_plane_image_1d_of_planes_from_convolver_image(
        self, convolver_image
    ):
        """Extract the 1D image-plane image and 1D blurring image-plane image of every plane and blur each with the \
        PSF using a convolver (see ccd.convolution).

        The blurred image of every plane is returned in 1D.

        Parameters
        ----------
        convolver_image : hyper.ccd.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """
        return [
            plane.blurred_profile_image_plane_image_1d_from_convolver_image(
                convolver_image=convolver_image
            )
            for plane in self.planes
        ]

    def unmasked_blurred_profile_image_plane_image_from_psf(self, psf):

        padded_tracer = self.padded_tracer_from_psf_shape(psf_shape=psf.shape)

        padded_image_1d = padded_tracer.profile_image_plane_image(
            return_in_2d=False, return_binned=True
        )

        return padded_tracer.grid_stack.unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
            padded_array_1d=padded_image_1d,
            psf=psf,
            image_shape=self.grid_stack.regular.mask.shape,
        )

    def unmasked_blurred_profile_image_plane_image_of_planes_from_psf(self, psf):

        unmasked_blurred_profile_image_plane_image_of_planes = []

        padded_tracer = self.padded_tracer_from_psf_shape(psf_shape=psf.shape)

        for padded_plane in padded_tracer.planes:

            padded_image_1d = padded_plane.profile_image_plane_image(
                return_in_2d=False, return_binned=True
            )

            unmasked_blurred_array_2d = padded_tracer.grid_stack.unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
                padded_array_1d=padded_image_1d,
                psf=psf,
                image_shape=self.grid_stack.regular.mask.shape,
            )

            unmasked_blurred_profile_image_plane_image_of_planes.append(
                unmasked_blurred_array_2d
            )

        return unmasked_blurred_profile_image_plane_image_of_planes

    def unmasked_blurred_profile_image_plane_image_of_plane_and_galaxies_from_psf(
        self, psf
    ):

        unmasked_blurred_profile_image_plane_image_of_planes_and_galaxies = []

        padded_tracer = self.padded_tracer_from_psf_shape(psf_shape=psf.shape)

        for padded_plane in padded_tracer.planes:

            padded_image_1d_of_galaxies = padded_plane.profile_image_plane_image_of_galaxies(
                return_in_2d=False, return_binned=True
            )

            unmasked_blurred_array_2d_of_galaxies = list(
                map(
                    lambda padded_image_1d_of_galaxy: padded_tracer.grid_stack.unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
                        padded_array_1d=padded_image_1d_of_galaxy,
                        psf=psf,
                        image_shape=self.grid_stack.regular.mask.shape,
                    ),
                    padded_image_1d_of_galaxies,
                )
            )

            unmasked_blurred_profile_image_plane_image_of_planes_and_galaxies.append(
                unmasked_blurred_array_2d_of_galaxies
            )

        return unmasked_blurred_profile_image_plane_image_of_planes_and_galaxies

    def inversion_from_image_1d_noise_map_1d_and_convolver_mapping_matrix(
        self, image_1d, noise_map_1d, convolver_mapping_matrix
    ):

        if len(self.mappers_of_planes) > 1:
            raise exc.RayTracingException(
                "PyAutoLens does not currently support more than one mapper, reglarization and"
                "therefore inversion per tracer."
            )

        return inv.Inversion.from_data_1d_mapper_and_regularization(
            image_1d=image_1d,
            noise_map_1d=noise_map_1d,
            convolver=convolver_mapping_matrix,
            mapper=self.mappers_of_planes[0],
            regularization=self.regularizations_of_planes[0],
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

    @property
    def galaxy_image_dict_blank_images(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_image_dict = dict()

        for plane in self.planes:
            for galaxy in plane.galaxies:

                galaxy_image_dict[galaxy] = plane.profile_image_plane_image_of_galaxy(
                    galaxy=galaxy, return_in_2d=False, return_binned=True
                )

        return galaxy_image_dict

    def galaxy_image_dict_from_convolver_image(
        self, convolver_image
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_image_dict = dict()

        for plane in self.planes:
            for galaxy in plane.galaxies:

                profile_image_plane_image_1d = plane.profile_image_plane_image_of_galaxy(
                    galaxy=galaxy, return_in_2d=False, return_binned=True
                )

                profile_image_plane_blurring_image_1d = plane.profile_image_plane_blurring_image_of_galaxy(
                    galaxy=galaxy, return_in_2d=False
                )

                blurred_profile_image_pane_image_1d = convolver_image.convolve_image(
                    image_array=profile_image_plane_image_1d,
                    blurring_array=profile_image_plane_blurring_image_1d,
                )

                galaxy_image_dict[galaxy] = blurred_profile_image_pane_image_1d

        return galaxy_image_dict


class Tracer(AbstractTracerData):
    def __init__(self, planes, cosmology=cosmo.Planck15):
        """Ray-tracer for a lens system with any number of planes.

        The redshift of these planes are specified by the redshits of the galaxies; there is a unique plane redshift \
        for every unique galaxy redshift (galaxies with identical redshifts are put in the same plane).

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid-stack (see grid_stack.GridStack) which is used for ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid_stack : grid_stacks.GridStack
            The image-plane grid stack which is traced. (includes the regular-grid, sub-grid, blurring-grid, etc.).
        border : masks.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        super(Tracer, self).__init__(planes=planes, cosmology=cosmology)

    @classmethod
    def from_galaxies_and_image_plane_grid_stack(
        cls, galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15
    ):

        plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(
            galaxies=galaxies
        )

        if len(plane_redshifts) == 1:
            return cls.x1_plane_tracer_from_lens_galaxies_and_image_plane_grid_stack(
                lens_galaxies=galaxies,
                image_plane_grid_stack=image_plane_grid_stack,
                border=border,
                cosmology=cosmology,
            )

        galaxies_in_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=plane_redshifts
        )

        if len(plane_redshifts) == 2:
            return cls.x2_plane_tracer_from_lens_and_source_galaxies_and_image_plane_grid_stack(
                lens_galaxies=galaxies_in_planes[0],
                source_galaxies=galaxies_in_planes[1],
                image_plane_grid_stack=image_plane_grid_stack,
                border=border,
                cosmology=cosmology,
            )

        planes = []

        for plane_index in range(0, len(plane_redshifts)):

            compute_deflections = lens_util.compute_deflections_at_next_plane(
                plane_index=plane_index, total_planes=len(plane_redshifts)
            )

            new_grid_stack = image_plane_grid_stack

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=plane_redshifts[previous_plane_index],
                        redshift_1=plane_redshifts[plane_index],
                        redshift_final=plane_redshifts[-1],
                        cosmology=cosmology,
                    )

                    scaled_deflections_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
                        plane=planes[previous_plane_index],
                        scaling_factor=scaling_factor,
                    )

                    new_grid_stack = lens_util.grid_stack_from_deflections_stack(
                        grid_stack=new_grid_stack,
                        deflections_stack=scaled_deflections_stack,
                    )

            planes.append(
                pl.Plane(
                    galaxies=galaxies_in_planes[plane_index],
                    grid_stack=new_grid_stack,
                    border=border,
                    compute_deflections=compute_deflections,
                    cosmology=cosmology,
                )
            )

        return Tracer(planes=planes, cosmology=cosmology)

    @classmethod
    def x1_plane_tracer_from_lens_galaxies_and_image_plane_grid_stack(
        cls,
        lens_galaxies,
        image_plane_grid_stack,
        border=None,
        cosmology=cosmo.Planck15,
    ):
        image_plane = pl.Plane(
            galaxies=lens_galaxies,
            grid_stack=image_plane_grid_stack,
            border=border,
            compute_deflections=False,
            cosmology=cosmology,
        )

        return Tracer(planes=[image_plane], cosmology=cosmology)

    @classmethod
    def x2_plane_tracer_from_lens_and_source_galaxies_and_image_plane_grid_stack(
        cls,
        lens_galaxies,
        source_galaxies,
        image_plane_grid_stack,
        border=None,
        cosmology=cosmo.Planck15,
    ):
        image_plane = pl.Plane(
            galaxies=lens_galaxies,
            grid_stack=image_plane_grid_stack,
            border=border,
            compute_deflections=True,
            cosmology=cosmology,
        )

        source_plane_grid_stack = image_plane.trace_grid_stack_to_next_plane()

        source_plane = pl.Plane(
            galaxies=source_galaxies,
            grid_stack=source_plane_grid_stack,
            border=border,
            compute_deflections=False,
            cosmology=cosmology,
        )
        return Tracer(planes=[image_plane, source_plane], cosmology=cosmology)

    @classmethod
    def sliced_tracer_from_lens_line_of_sight_and_source_galaxies(
        cls,
        lens_galaxies,
        line_of_sight_galaxies,
        source_galaxies,
        planes_between_lenses,
        image_plane_grid_stack,
        border=None,
        cosmology=cosmo.Planck15,
    ):

        """Ray-tracer for a lens system with any number of planes.

        The redshift of these planes are specified by the input parameters *lens_redshifts* and \
         *slices_between_main_planes*. Every galaxy is placed in its closest plane in redshift-space.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid-stack (see grid_stack.GridStack) which is used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid_stack : grid_stacks.GridStack
            The image-plane grid stack which is traced. (includes the regular-grid, sub-grid, blurring-grid, etc.).
        planes_between_lenses : [int]
            The number of slices between each main plane. The first entry in this list determines the number of slices \
            between Earth (redshift 0.0) and main plane 0, the next between main planes 0 and 1, etc.
        border : masks.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        lens_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(
            galaxies=lens_galaxies
        )

        plane_redshifts = lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
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

            compute_deflections = lens_util.compute_deflections_at_next_plane(
                plane_index=plane_index, total_planes=len(plane_redshifts)
            )

            new_grid_stack = image_plane_grid_stack

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=plane_redshifts[previous_plane_index],
                        redshift_1=plane_redshifts[plane_index],
                        redshift_final=plane_redshifts[-1],
                        cosmology=cosmology,
                    )

                    scaled_deflections_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
                        plane=planes[previous_plane_index],
                        scaling_factor=scaling_factor,
                    )

                    new_grid_stack = lens_util.grid_stack_from_deflections_stack(
                        grid_stack=new_grid_stack,
                        deflections_stack=scaled_deflections_stack,
                    )

            planes.append(
                pl.Plane(
                    redshift=plane_redshifts[plane_index],
                    galaxies=galaxies_in_planes[plane_index],
                    grid_stack=new_grid_stack,
                    border=border,
                    compute_deflections=compute_deflections,
                    cosmology=cosmology,
                )
            )

        return Tracer(planes=planes, cosmology=cosmology)

    @classmethod
    def from_galaxies_and_image_plane_positions(
        cls, galaxies, image_plane_positions, cosmology=cosmo.Planck15
    ):

        plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(
            galaxies=galaxies
        )

        galaxies_in_planes = lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(
            galaxies=galaxies, plane_redshifts=plane_redshifts
        )

        planes = []

        for plane_index in range(0, len(plane_redshifts)):

            compute_deflections = lens_util.compute_deflections_at_next_plane(
                plane_index=plane_index, total_planes=len(plane_redshifts)
            )

            new_positions = image_plane_positions

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=plane_redshifts[previous_plane_index],
                        redshift_1=plane_redshifts[plane_index],
                        redshift_final=plane_redshifts[-1],
                        cosmology=cosmology,
                    )

                    scaled_deflections = list(
                        map(
                            lambda deflections: np.multiply(
                                scaling_factor, deflections
                            ),
                            planes[previous_plane_index].deflections,
                        )
                    )

                    new_positions = list(
                        map(
                            lambda positions, deflections: np.subtract(
                                positions, deflections
                            ),
                            new_positions,
                            scaled_deflections,
                        )
                    )

            planes.append(
                pl.PlanePositions(
                    redshift=plane_redshifts[plane_index],
                    galaxies=galaxies_in_planes[plane_index],
                    positions=new_positions,
                    compute_deflections=compute_deflections,
                )
            )

        return Tracer(planes=planes, cosmology=cosmology)

    def padded_tracer_from_psf_shape(self, psf_shape):

        padded_grid_stack = self.grid_stack.padded_grid_stack_from_psf_shape(
            psf_shape=psf_shape
        )

        return Tracer.from_galaxies_and_image_plane_grid_stack(
            galaxies=self.galaxies,
            image_plane_grid_stack=padded_grid_stack,
            border=self.image_plane.border,
            cosmology=self.cosmology,
        )
