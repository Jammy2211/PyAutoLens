import math

import numpy as np
from astropy import cosmology as cosmo
from functools import wraps

from autolens import exc
from autolens.model import cosmology_util
from autolens.data.array import grids
from autolens.lens import plane as pl
from autolens.lens.util import lens_util
from autolens.model.inversion import pixelizations as pix


def check_tracer_for_light_profile(func):
    """If none of the tracer's galaxies have a light profile, it image-plane image cannot be computed. This wrapper \
    makes this property return *None*.

    Parameters
    ----------
    func : (self) -> Object
        A property function that requires galaxies to have a mass profile.
    """

    @wraps(func)
    def wrapper(self):
        """

        Parameters
        ----------
        self

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        if self.has_light_profile is True:
            return func(self)
        else:
            return None

    return wrapper


def check_tracer_for_mass_profile(func):
    """If none of the tracer's galaxies have a mass profile, it surface density, potential and deflections cannot \
    be computed. This wrapper makes these properties return *None*.

    Parameters
    ----------
    func : (self) -> Object
        A property function that requires galaxies to have a mass profile.
    """

    @wraps(func)
    def wrapper(self):
        """

        Parameters
        ----------
        self

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        if self.has_mass_profile is True:
            return func(self)
        else:
            return None

    return wrapper


class AbstractTracerCosmology(object):

    def __init__(self, plane_redshifts, cosmology):
        """Abstract Ray tracer for lens systems with any number of planes.

        From the galaxies of the tracer's planes, their grid-stack(s) and the cosmology physically derived quantities \
        (e.g. surface density, angular diameter distances, critical surface densities) can be computed.

        Parameters
        ----------
        plane_redshifts : [pl.Plane] or [pl.PlaneStack]
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
        return cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(redshift=self.plane_redshifts[i],
                                                                         cosmology=self.cosmology)

    def kpc_per_arcsec_proper_of_plane(self, i):
        return 1.0 / self.arcsec_per_kpc_proper_of_plane(i=i)

    def angular_diameter_distance_of_plane_to_earth_in_units(self, i, unit_length='arcsec'):
        return cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
            redshift=self.plane_redshifts[i], cosmology=self.cosmology, unit_length=unit_length)

    def angular_diameter_distance_between_planes_in_units(self, i, j, unit_length='arcsec'):
        return cosmology_util.angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(
            redshift_0=self.plane_redshifts[i], redshift_1=self.plane_redshifts[j], cosmology=self.cosmology,
            unit_length=unit_length)

    def angular_diameter_distance_to_source_plane_in_units(self, unit_length='arcsec'):
        return cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
            redshift=self.plane_redshifts[-1], cosmology=self.cosmology, unit_length=unit_length)

    def critical_surface_density_between_planes_in_units(self, i, j, unit_length='arcsec', unit_mass='solMass'):
        return cosmology_util.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
            redshift_0=self.plane_redshifts[i], redshift_1=self.plane_redshifts[j], cosmology=self.cosmology,
            unit_length=unit_length, unit_mass=unit_mass)

    def scaling_factor_between_planes(self, i, j):
        return cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
            redshift_0=self.plane_redshifts[i],
            redshift_1=self.plane_redshifts[j], redshift_final=self.plane_redshifts[-1], cosmology=self.cosmology)

    def angular_diameter_distance_from_image_to_source_plane_in_units(self, unit_length='arcsec'):
        return self.angular_diameter_distance_between_planes_in_units(i=0, j=-1, unit_length=unit_length)


class AbstractTracer(AbstractTracerCosmology):

    def __init__(self, planes, cosmology):
        """Abstract Ray tracer for lens systems with any number of planes.

        From the galaxies of the tracer's planes, their grid-stack(s) and the cosmology physically derived quantities \
        (e.g. surface density, angular diameter distances, critical surface densities) can be computed.

        Parameters
        ----------
        planes : [pl.Plane] or [pl.PlaneStack]
            The list of the tracer's planes in ascending redshift order.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        self.planes = planes
        super(AbstractTracer, self).__init__(plane_redshifts=[plane.redshift for plane in self.planes],
                                             cosmology=cosmology)

    def plane_with_galaxy(self, galaxy):
        return [plane for plane in self.planes if galaxy in plane.galaxies][0]

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
    def has_padded_grids(self):
        return isinstance(self.planes[0].grids.regular, grids.PaddedRegularGrid)

    @property
    def galaxies(self):
        return list([galaxy for plane in self.planes for galaxy in plane.galaxies])

    @property
    def galaxies_in_planes(self):
        return list([plane.galaxies for plane in self.planes])

    @property
    @check_tracer_for_light_profile
    def image_plane_image(self):
        return self.image_plane.grid_stack.scaled_array_2d_from_array_1d(array_1d=self.image_plane_image_1d)

    @property
    @check_tracer_for_light_profile
    def image_plane_image_for_simulation(self):
        return sum(self.image_plane_image_of_planes_for_simulation)

    @property
    def image_plane_image_of_planes_for_simulation(self):
        return [plane.image_plane_image_for_simulation for plane in self.planes]

    @property
    @check_tracer_for_light_profile
    def image_plane_image_1d(self):
        return sum(self.image_plane_image_1d_of_planes)

    @property
    def image_plane_image_1d_of_planes(self):
        return [plane.image_plane_image_1d for plane in self.planes]

    @property
    @check_tracer_for_light_profile
    def image_plane_blurring_image_1d(self):
        return sum(self.image_plane_blurring_image_1d_of_planes)

    @property
    def image_plane_blurring_image_1d_of_planes(self):
        return [plane.image_plane_blurring_image_1d for plane in self.planes]

    @property
    def mappers_of_planes(self):
        return list(filter(None, [plane.mapper for plane in self.planes]))

    @property
    def regularizations_of_planes(self):
        return list(filter(None, [plane.regularization for plane in self.planes]))

    @property
    @check_tracer_for_mass_profile
    def convergence(self):
        return sum([plane.convergence for plane in self.planes])

    @property
    @check_tracer_for_mass_profile
    def potential(self):
        return sum([plane.potential for plane in self.planes])

    @property
    @check_tracer_for_mass_profile
    def deflections_y(self):
        return sum([plane.deflections_y for plane in self.planes])

    @property
    @check_tracer_for_mass_profile
    def deflections_x(self):
        return sum([plane.deflections_x for plane in self.planes])

    def einstein_radius_of_plane_in_units(self, i, unit_length='arcsec'):
        return self.planes[i].einstein_radius_in_units(unit_length=unit_length)

    def einstein_mass_between_planes_in_units(self, i, j, unit_mass='solMass'):
        return self.planes[i].einstein_mass_in_units(unit_mass=unit_mass, redshift_source=self.plane_redshifts[j])

    def grid_at_redshift_from_image_plane_grid_and_redshift(self, image_plane_grid, redshift):
        """For an input grid of (y,x) arc-second image-plane coordinates, ray-trace the coordinates to any redshift in \
        the strong lens configuration.

        This is performed using multi-plane ray-tracing and the existing redshifts and planes of the tracer. However, \
        any redshift can be input even if a plane does not exist there, including redshifts before the first plane \
        of the lensing system.

        Parameters
        ----------
        image_plane_grid : ndsrray or grids.RegularGrid
            The image-plane grid which is traced to the redshift.
        redshift : float
            The redshift the image-plane grid is traced to.
        """

        # TODO : We need to come up with a better abstraction for multi-plane lensing 0_0

        image_plane_grid_stack = grids.GridStack(regular=image_plane_grid, sub=np.array([[0.0, 0.0]]),
                                                 blurring=np.array([[0.0, 0.0]]))

        tracer = TracerMultiPlanes(galaxies=self.galaxies, image_plane_grid_stack=image_plane_grid_stack,
                                   border=None, cosmology=self.cosmology)

        for plane_index in range(0, len(self.plane_redshifts)):

            new_grid_stack = image_plane_grid_stack

            if redshift <= tracer.plane_redshifts[plane_index]:

                # If redshift is between two planes, we need to map over all previous planes coordinates / deflections.

                if plane_index > 0:
                    for previous_plane_index in range(plane_index):
                        scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                            redshift_0=tracer.plane_redshifts[previous_plane_index], redshift_1=redshift,
                            redshift_final=tracer.plane_redshifts[-1], cosmology=tracer.cosmology)

                        scaled_deflection_stack = lens_util.scaled_deflection_stack_from_plane_and_scaling_factor(
                            plane=tracer.planes[previous_plane_index], scaling_factor=scaling_factor)

                        new_grid_stack = \
                            lens_util.grid_stack_from_deflection_stack(grid_stack=new_grid_stack,
                                                                       deflection_stack=scaled_deflection_stack)

                # If redshift is before the first plane, no change to image pllane coordinates.

                elif plane_index == 0:

                    return new_grid_stack.regular

                return new_grid_stack.regular


class TracerImagePlane(AbstractTracer):

    def __init__(self, lens_galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
        """Ray tracer for a lens system with just an image-plane. 
        
        As there is only 1 plane, there are no ray-tracing calculations. This class is therefore only used for fitting \ 
        image-plane galaxies with light profiles.
        
        This tracer has only one grid-stack (see grid_stack.GridStack) which is used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        image_plane_grid_stack : grid_stacks.GridStack
            The image-plane grid stack which is traced. (includes the regular-grid, sub-grid, blurring-grid, etc.).
        border : masks.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        if not lens_galaxies:
            raise exc.RayTracingException('No lens galaxies have been input into the Tracer')

        image_plane = pl.Plane(galaxies=lens_galaxies, grid_stack=image_plane_grid_stack, border=border,
                               compute_deflections=True, cosmology=cosmology)

        super(TracerImagePlane, self).__init__(planes=[image_plane], cosmology=cosmology)

    def critical_surface_density_between_planes_in_units(self, i, j, unit_length='arcsec', unit_mass='solMass'):
        return 0.0


class TracerImageSourcePlanes(AbstractTracer):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lens system with two planes, an image-plane and source-plane.

        This tracer has only one grid-stack (see grid_stack.GridStack) which is used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of galaxies in the source-plane.
        image_plane_grid_stack : grid_stacks.GridStack
            The image-plane grid stack which is traced. (includes the regular-grid, sub-grid, blurring-grid, etc.).
        border : masks.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        image_plane_grid_stack = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
            galaxies=source_galaxies, grid_stack=image_plane_grid_stack)

        image_plane = pl.Plane(galaxies=lens_galaxies, grid_stack=image_plane_grid_stack, border=border,
                               compute_deflections=True, cosmology=cosmology)

        source_plane_grid_stack = image_plane.trace_grid_stack_to_next_plane()

        source_plane = pl.Plane(galaxies=source_galaxies, grid_stack=source_plane_grid_stack, border=border,
                                compute_deflections=False, cosmology=cosmology)

        super(TracerImageSourcePlanes, self).__init__(planes=[image_plane, source_plane], cosmology=cosmology)

    def critical_surface_density_between_image_and_source_plane_in_units(self, unit_length='arcsec', unit_mass='solMass'):
        return self.critical_surface_density_between_planes_in_units(i=0, j=1, unit_length=unit_length,
                                                                     unit_mass=unit_mass)

    def einstein_radius_of_image_plane_in_units(self, unit_length='arcsec'):
        return self.einstein_radius_of_plane_in_units(i=0, unit_length=unit_length)

    def einstein_mass_between_image_and_source_plane_in_units(self, unit_mass='solMass'):
        return self.einstein_mass_between_planes_in_units(i=0, j=1, unit_mass=unit_mass)


class TracerMultiPlanes(AbstractTracer):

    def __init__(self, galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
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

        plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        galaxies_in_planes = \
            lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=galaxies,
                                                                        plane_redshifts=plane_redshifts)

        image_plane_grid_stack = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
            galaxies=galaxies, grid_stack=image_plane_grid_stack)

        planes = []

        for plane_index in range(0, len(plane_redshifts)):

            compute_deflections = lens_util.compute_deflections_at_next_plane(plane_index=plane_index,
                                                                              total_planes=len(plane_redshifts))

            new_grid_stack = image_plane_grid_stack

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=plane_redshifts[previous_plane_index], redshift_1=plane_redshifts[plane_index],
                        redshift_final=plane_redshifts[-1], cosmology=cosmology)

                    scaled_deflection_stack = lens_util.scaled_deflection_stack_from_plane_and_scaling_factor(
                        plane=planes[previous_plane_index], scaling_factor=scaling_factor)

                    new_grid_stack = \
                        lens_util.grid_stack_from_deflection_stack(grid_stack=new_grid_stack,
                                                                   deflection_stack=scaled_deflection_stack)

            planes.append(pl.Plane(galaxies=galaxies_in_planes[plane_index], grid_stack=new_grid_stack,
                                   border=border, compute_deflections=compute_deflections, cosmology=cosmology))

        super(TracerMultiPlanes, self).__init__(planes=planes, cosmology=cosmology)


class TracerMultiPlanesSliced(AbstractTracer):

    def __init__(self, lens_galaxies, line_of_sight_galaxies, source_galaxies, planes_between_lenses,
                 image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
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

        lens_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=lens_galaxies)

        plane_redshifts = lens_util.ordered_plane_redshifts_from_lens_and_source_plane_redshifts_and_slice_sizes(
            lens_redshifts=lens_redshifts, planes_between_lenses=planes_between_lenses,
            source_plane_redshift=source_galaxies[0].redshift)

        galaxies_in_planes = \
            lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=lens_galaxies + line_of_sight_galaxies,
                                                                        plane_redshifts=plane_redshifts)

        plane_redshifts.append(source_galaxies[0].redshift)
        galaxies_in_planes.append(source_galaxies)

        image_plane_grid_stack = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
            galaxies=lens_galaxies, grid_stack=image_plane_grid_stack)

        planes = []

        for plane_index in range(0, len(plane_redshifts)):

            compute_deflections = lens_util.compute_deflections_at_next_plane(plane_index=plane_index,
                                                                              total_planes=len(plane_redshifts))

            new_grid_stack = image_plane_grid_stack

            if plane_index > 0:
                print()
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=plane_redshifts[previous_plane_index], redshift_1=plane_redshifts[plane_index],
                        redshift_final=plane_redshifts[-1], cosmology=cosmology)

                    scaled_deflection_stack = lens_util.scaled_deflection_stack_from_plane_and_scaling_factor(
                        plane=planes[previous_plane_index], scaling_factor=scaling_factor)

                    new_grid_stack = \
                        lens_util.grid_stack_from_deflection_stack(grid_stack=new_grid_stack,
                                                                   deflection_stack=scaled_deflection_stack)

            planes.append(pl.PlaneSlice(redshift=plane_redshifts[plane_index], galaxies=galaxies_in_planes[plane_index],
                                        grid_stack=new_grid_stack, border=border,
                                        compute_deflections=compute_deflections, cosmology=cosmology))

        super(TracerMultiPlanesSliced, self).__init__(planes=planes, cosmology=cosmology)


class TracerImageSourcePlanesPositions(AbstractTracer):

    def __init__(self, lens_galaxies, image_plane_positions, cosmology=cosmo.Planck15):
        """Positional ray-tracer for a lens system with two planes, an image-plane and source-plane (source-plane \
        galaxies are not input for the positional ray-tracer, as it is only the proximity that image_plane_positions \
        trace to within one another that needs to be computed).

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. If a cosmology is supplied, the plane's angular diameter distances, \
        conversion factors, etc. are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        image_plane_positions : [[[]]]
            The (y,x) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same \
            location(s) in the source-plane.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        image_plane = pl.PlanePositions(redshift=lens_galaxies[0].redshift, galaxies=lens_galaxies,
                                        positions=image_plane_positions, compute_deflections=True, cosmology=cosmology)

        source_plane_positions = image_plane.trace_to_next_plane()

        source_plane = pl.PlanePositions(redshift=None, galaxies=None, positions=source_plane_positions,
                                         compute_deflections=False, cosmology=cosmology)

        super(TracerImageSourcePlanesPositions, self).__init__(planes=[image_plane, source_plane], cosmology=cosmology)


class TracerMultiPlanesPositions(AbstractTracer):

    def __init__(self, galaxies, image_plane_positions, cosmology=cosmo.Planck15):
        """Positional ray-tracer for a lens system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_positions : [[[]]]
            The (y,x) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same \
            location(s) in the final source-plane.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        plane_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        galaxies_in_redshift_ordered_lists = \
            lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=galaxies,
                                                                        plane_redshifts=plane_redshifts)

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        planes = []

        for plane_index in range(0, len(plane_redshifts)):

            if plane_index < len(plane_redshifts) - 1:
                compute_deflections = True
            elif plane_index == len(plane_redshifts) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_positions = image_plane_positions

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                        redshift_0=plane_redshifts[previous_plane_index], redshift_1=plane_redshifts[plane_index],
                        redshift_final=plane_redshifts[-1], cosmology=cosmology)

                    scaled_deflections = list(map(lambda deflections:
                                                  np.multiply(scaling_factor, deflections),
                                                  planes[previous_plane_index].deflections))

                    new_positions = list(map(lambda positions, deflections:
                                             np.subtract(positions, deflections), new_positions, scaled_deflections))

            planes.append(pl.PlanePositions(redshift=plane_redshifts[plane_index],
                                            galaxies=galaxies_in_redshift_ordered_lists[plane_index],
                                            positions=new_positions, compute_deflections=compute_deflections))

        super(TracerMultiPlanesPositions, self).__init__(planes=planes, cosmology=cosmology)
