from functools import wraps
import math

import numpy as np
from astropy import constants
from astropy import cosmology as cosmo

from autolens import exc
from autolens.data.array import grids
from autolens.lensing import lensing_util
from autolens.lensing import plane as pl
from autolens.model.inversion import pixelizations as pix

def check_tracer_for_redshifts(func):
    """If a tracer's galaxies do not have redshifts, its cosmological quantities cannot be computed. This wrapper \
    makes these functions return *None* if the galaxies do not have redshifts

    Parameters
    ----------
    func : (self) -> Object
        A property function that requires galaxies to have redshifts.
    """

    @wraps(func)
    def wrapper(self):
        """

        Parameters
        ----------
        self
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        if self.all_planes_have_redshifts is True:
            return func(self)
        else:
            return None

    return wrapper


class AbstractTracer(object):

    def __init__(self, planes, cosmology):
        """Abstract Ray tracer for lensing systems with any number of planes.

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
        self.cosmology = cosmology

    @property
    def image_plane(self):
        return self.planes[0]

    @property
    def source_plane(self):
        return self.planes[-1]

    @property
    def total_planes(self):
        return len(self.planes)

    @property
    def plane_redshifts(self):
        return [plane.redshift for plane in self.planes]

    @property
    def all_planes_have_redshifts(self):
        return not None in self.plane_redshifts

    @property
    def has_light_profile(self):
        return any(list(map(lambda plane: plane.has_light_profile, self.planes)))

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
    def has_hyper_galaxy(self):
        return any(list(map(lambda plane: plane.has_hyper_galaxy, self.planes)))

    @property
    def hyper_galaxies(self):
        return list(filter(None, [hyper_galaxy for plane in self.planes for hyper_galaxy in plane.hyper_galaxies]))

    @property
    def mappers_of_planes(self):
        return list(filter(None, [plane.mapper for plane in self.planes]))

    @property
    def regularizations_of_planes(self):
        return list(filter(None, [plane.regularization for plane in self.planes]))

    @property
    def surface_density(self):
        return sum([plane.surface_density for plane in self.planes])

    @property
    def potential(self):
        return sum([plane.potential for plane in self.planes])

    @property
    def deflections_y(self):
        return sum([plane.deflections_y for plane in self.planes])

    @property
    def deflections_x(self):
        return sum([plane.deflections_x for plane in self.planes])

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    def arcsec_per_kpc_proper_of_plane(self, i):
        return self.cosmology.arcsec_per_kpc_proper(z=self.plane_redshifts[i]).value

    def kpc_per_arcsec_proper_of_plane(self, i):
        return 1.0 / self.arcsec_per_kpc_proper_of_plane(i)

    def angular_diameter_distance_of_plane_to_earth(self, i):
        return self.cosmology.angular_diameter_distance(self.plane_redshifts[i]).to('kpc').value

    def angular_diameter_distance_between_planes(self, i, j):
        return self.cosmology.angular_diameter_distance_z1z2(self.plane_redshifts[i],
                                                             self.plane_redshifts[j]).to('kpc').value

    @property
    def angular_diameter_distance_to_source_plane(self):
        return self.cosmology.angular_diameter_distance(self.plane_redshifts[-1]).to('kpc').value

    def critical_density_kpc_between_planes(self, i, j):
        return self.constant_kpc * self.angular_diameter_distance_of_plane_to_earth(j) / \
               (self.angular_diameter_distance_between_planes(i, j) *
                self.angular_diameter_distance_of_plane_to_earth(i))

    def critical_density_arcsec_between_planes(self, i, j):
        return self.critical_density_kpc_between_planes(i=i, j=j) * self.kpc_per_arcsec_proper_of_plane(i=i) ** 2.0

    def scaling_factor_between_planes(self, i, j):
        return lensing_util.scaling_factor_between_redshifts_for_cosmology(z1=self.plane_redshifts[i], z2=self.plane_redshifts[j],
                                                              z_final=self.plane_redshifts[-1], cosmology=self.cosmology)

    @property
    @check_tracer_for_redshifts
    def angular_diameter_distance_from_image_to_source_plane(self):
        return self.angular_diameter_distance_between_planes(i=0, j=-1)

    @property
    @check_tracer_for_redshifts
    def critical_density_kpc(self):
        return self.constant_kpc * self.source_plane.angular_diameter_distance_to_earth / \
               (self.angular_diameter_distance_from_image_to_source_plane *
                self.image_plane.angular_diameter_distance_to_earth)

    @property
    @check_tracer_for_redshifts
    def critical_density_arcsec(self):
        return self.critical_density_kpc * self.image_plane.kpc_per_arcsec_proper ** 2.0

    def masses_of_image_plane_galaxies_within_circles(self, radius):
        """
        Compute the total mass of all galaxies in the image-plane within a circle of specified radius, using the \ 
        plane's critical surface density to convert this to physical units.

        For a single galaxy, inputting the Einstein Radius should provide an accurate measurement of the Einstein \ 
        mass. Use of other radii may be subject to systematic offsets, because lensing does not directly measure the \
        mass of a galaxy beyond the Einstein radius.

        For multiple galaxies, the Einstein mass of the entire image-plane is evenly divided across its galaxies. \ 
        This could be highly inaccurate and users are recommended to cross-check mass estimates using different radii.

        See *galaxy.dimensionless_mass_within_circle* and *mass_profiles.dimensionless_mass_within_circle* for details \
        of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        """
        if self.cosmology is not None:
            return self.image_plane.masses_of_galaxies_within_circles(radius=radius,
                                                                      conversion_factor=self.critical_density_arcsec)
        else:
            return None

    def masses_of_image_plane_galaxies_within_ellipses(self, major_axis):
        """
        Compute the total mass of all galaxies in this plane within a ellipse of specified major-axis.

        For a single galaxy, inputting the Einstein Radius should provide an accurate measurement of the Einstein \ 
        mass. Use of other radii may be subject to systematic offsets, because lensing does not directly measure the \
        mass of a galaxy beyond the Einstein radius.

        For multiple galaxies, the Einstein mass of the entire image-plane is evenly divided across its galaxies. \ 
        This could be highly inaccurate and users are recommended to cross-check mass estimates using different radii.

        See *galaxy.dimensionless_mass_within_ellipse* and *mass_profiles.dimensionless_mass_within_ellipse* for details
        of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major-axis of the ellipse to compute the dimensionless mass within.
        """
        if self.cosmology is not None:
            return self.image_plane.masses_of_galaxies_within_ellipses(major_axis=major_axis,
                                                                       conversion_factor=self.critical_density_arcsec)
        else:
            return None


class AbstractTracerNonStack(AbstractTracer):

    def __init__(self, planes, cosmology):
        """Abstract ray-tracer for lensing systems with any number of planes and just one grid-stack.

        From the galaxies in the tracer's planes, their image-plane images are computed.

        Parameters
        ----------
        planes : [pl.Plane]
            The list of the tracer's planes in ascending redshift order.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """
        super(AbstractTracerNonStack, self).__init__(planes=planes, cosmology=cosmology)

    @property
    def image_plane_image(self):
        return  self.image_plane.grid_stack.regular.scaled_array_from_array_1d(self.image_plane_image_1d)

    @property
    def image_plane_image_for_simulation(self):
        return sum(self.image_plane_image_of_planes_for_simulation)

    @property
    def image_plane_image_of_planes_for_simulation(self):
        return [plane.image_plane_image_for_simulation for plane in self.planes]

    @property
    def image_plane_image_1d(self):
        return sum(self.image_plane_image_1d_of_planes)

    @property
    def image_plane_image_1d_of_planes(self):
        return [plane.image_plane_image_1d for plane in self.planes]

    @property
    def image_plane_blurring_image_1d(self):
        return sum(self.image_plane_blurring_image_of_planes_1d)

    @property
    def image_plane_blurring_image_of_planes_1d(self):
        return [plane.image_plane_blurring_image_1d for plane in self.planes]


class AbstractTracerStack(AbstractTracer):

    def __init__(self, planes, cosmology):
        """Abstract ray-tracer for lensing systems with any number of planes and multiple grid-stacks.

        From the galaxies in the tracer's planes, the image-plane images of each grid-stack are computed.

        Parameters
        ----------
        planes : [pl.Plane]
            The list of the tracer's planes in ascending redshift order.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """
        super(AbstractTracerStack, self).__init__(planes=planes, cosmology=cosmology)

    @property
    def total_grid_stacks(self):
        return len(self.image_plane.grid_stacks)

    @property
    def image_plane_images(self):
        return list(map(lambda image_plane_image_1d, grid_stack :
                        grid_stack.regular.scaled_array_from_array_1d(image_plane_image_1d),
                        self.image_plane_images_1d, self.image_plane.grid_stacks))

    @property
    def image_plane_images_for_simulation(self):
        return[sum(image_plane_image_of_plane_for_simulation[i] for image_plane_image_of_plane_for_simulation in
                   self.image_plane_images_of_planes_for_simulation) for i in range(self.total_grid_stacks)]

    @property
    def image_plane_images_of_planes_for_simulation(self):
        return [plane.image_plane_images_for_simulation for plane in self.planes]

    @property
    def image_plane_images_1d(self):
        return list(map(lambda image_plane_image_1d_of_planes : sum(image_plane_image_1d_of_planes),
                        self.image_plane_images_1d_of_planes))

    @property
    def image_plane_images_1d_of_planes(self):
        image_plane_images_1d = [plane.image_plane_images_1d for plane in self.planes]
        image_plane_images_1d_of_planes = [[] for _ in range(self.total_grid_stacks)]
        for image_index in range(self.total_grid_stacks):
            for plane_index in range(self.total_planes):
                image_plane_images_1d_of_planes[image_index].append(image_plane_images_1d[plane_index][image_index])
        return image_plane_images_1d_of_planes

    @property
    def image_plane_blurring_images_1d(self):
        return list(map(lambda image_plane_blurring_image_1d_of_planes : sum(image_plane_blurring_image_1d_of_planes),
                        self.image_plane_blurring_images_of_planes_1d))

    @property
    def image_plane_blurring_images_of_planes_1d(self):
        image_plane_blurring_images_1d = [plane.image_plane_blurring_images_1d for plane in self.planes]
        image_plane_blurring_images_1d_of_planes = [[] for _ in range(self.total_grid_stacks)]
        for image_index in range(self.total_grid_stacks):
            for plane_index in range(self.total_planes):
                image_plane_blurring_images_1d_of_planes[image_index].append(
                    image_plane_blurring_images_1d[plane_index][image_index])
        return image_plane_blurring_images_1d_of_planes


class TracerImagePlane(AbstractTracerNonStack):

    def __init__(self, lens_galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
        """Ray tracer for a lensing system with just an image-plane. 
        
        As there is only 1 plane, there are no ray-tracing calculations. This class is therefore only used for fitting \ 
        image-plane galaxies with light profiles.
        
        This tracer has only one grid-stack (see grids.GridStack) which is used for ray-tracing.

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


class TracerImagePlaneStack(AbstractTracerStack):

    def __init__(self, lens_galaxies, image_plane_grid_stacks, borders=None, cosmology=cosmo.Planck15):
        """Ray tracer for a lensing system with just an image-plane.

        As there is only 1 plane, there are no ray-tracing calculations. This class is therefore only used for fitting \
        image-plane galaxies with light profiles.

        This tracer has a list of grid-stacks (see grids.GridStack) which are all used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        image_plane_grid_stacks : [grid_stacks.GridStack]
            The image-plane grid stacks which are traced. (each stack includes the regular-grid, sub-grid, \
            blurring-grid, etc.).
        borders : [masks.RegularGridBorder]
            The border of each grid-stacks's regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane border.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        if not lens_galaxies:
            raise exc.RayTracingException('No lens galaxies have been input into the Tracer')

        image_plane = pl.PlaneStack(galaxies=lens_galaxies, grid_stacks=image_plane_grid_stacks, borders=borders,
                               compute_deflections=True, cosmology=cosmology)

        super(TracerImagePlaneStack, self).__init__(planes=[image_plane], cosmology=cosmology)


class TracerImageSourcePlanes(AbstractTracerNonStack):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lensing system with two planes, an image-plane and source-plane.

        This tracer has only one grid-stack (see grids.GridStack) which is used for ray-tracing.

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

        source_plane_grid_stack = image_plane.trace_grids_to_next_plane()

        source_plane = pl.Plane(galaxies=source_galaxies, grid_stack=source_plane_grid_stack, border=border,
                                compute_deflections=False, cosmology=cosmology)

        super(TracerImageSourcePlanes, self).__init__(planes=[image_plane, source_plane], cosmology=cosmology)


class TracerImageSourcePlanesStack(AbstractTracerStack):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grid_stacks, borders=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lensing system with two planes, an image-plane and source-plane.

        This tracer has a list of grid-stacks (see grids.GridStack) which are all used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of galaxies in the source-plane.
        image_plane_grid_stacks : [grid_stacks.GridStack]
            The image-plane grid stacks which are traced. (each stack includes the regular-grid, sub-grid, \
            blurring-grid, etc.).
        borders : [masks.RegularGridBorder]
            The border of each grid-stacks's regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane border.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        image_plane_grid_stacks = list(map(lambda data_grids :
                        pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(galaxies=source_galaxies,
                                                                                             grid_stack=data_grids),
                                           image_plane_grid_stacks))

        image_plane = pl.PlaneStack(galaxies=lens_galaxies, grid_stacks=image_plane_grid_stacks, borders=borders,
                               compute_deflections=True)

        source_plane_grid_stacks = image_plane.trace_grids_to_next_plane()

        source_plane = pl.PlaneStack(galaxies=source_galaxies, grid_stacks=source_plane_grid_stacks,
                                     borders=borders, compute_deflections=False, cosmology=cosmology)

        super(TracerImageSourcePlanesStack, self).__init__(planes=[image_plane, source_plane], cosmology=cosmology)


class TracerMultiPlanes(AbstractTracerNonStack):

    def __init__(self, galaxies, image_plane_grid_stack, border=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lensing-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid-stack (see grids.GridStack) which is used for ray-tracing.

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

        ordered_redshifts = lensing_util.ordered_redshifts_from_galaxies(galaxies=galaxies)

        galaxies_in_redshift_ordered_lists = \
            lensing_util.galaxies_in_redshift_ordered_lists_from_galaxies(galaxies=galaxies,
                                                                           ordered_redshifts=ordered_redshifts)

        image_plane_grid_stack = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
            galaxies=galaxies, grid_stack=image_plane_grid_stack)

        planes = []

        for plane_index in range(0, len(ordered_redshifts)):

            if plane_index < len(ordered_redshifts) - 1:
                compute_deflections = True
            elif plane_index == len(ordered_redshifts) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_grid_stack = image_plane_grid_stack

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = lensing_util.scaling_factor_between_redshifts_for_cosmology(
                        z1=ordered_redshifts[previous_plane_index], z2=ordered_redshifts[plane_index],
                        z_final=ordered_redshifts[-1], cosmology=cosmology)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    if planes[previous_plane_index].deflection_stack is not None:
                        scaled_deflections = planes[previous_plane_index].deflection_stack.apply_function(scale)
                    else:
                        scaled_deflections = None

                    if scaled_deflections is not None:

                        def minus(grid, deflections):
                            return grid - deflections

                        new_grid_stack = new_grid_stack.map_function(minus, scaled_deflections)

            planes.append(pl.Plane(galaxies=galaxies_in_redshift_ordered_lists[plane_index], grid_stack=new_grid_stack,
                                   border=border, compute_deflections=compute_deflections, cosmology=cosmology))

        super(TracerMultiPlanes, self).__init__(planes=planes, cosmology=cosmology)


class TracerMultiPlanesStack(AbstractTracerStack):

    def __init__(self, galaxies, image_plane_grid_stacks, borders=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, the cosmology that is input is used to rescale deflection-angles \
        according to the lensing-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has a list of grid-stacks (see grids.GridStack) which are all used for ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid_stacks : [grid_stacks.GridStack]
            The image-plane grid stacks which are traced. (each stack includes the regular-grid, sub-grid, \
            blurring-grid, etc.).
        borders : [masks.RegularGridBorder]
            The border of each grid-stacks's regular-grid, which is used to relocate demagnified traced pixels to the \
            source-plane border.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        ordered_redshifts = lensing_util.ordered_redshifts_from_galaxies(galaxies=galaxies)

        galaxies_in_redshift_ordered_lists = \
            lensing_util.galaxies_in_redshift_ordered_lists_from_galaxies(galaxies=galaxies,
                                                                          ordered_redshifts=ordered_redshifts)

        image_plane_grid_stacks = list(map(lambda grid_stack :
                        pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(galaxies=galaxies,
                                                                                             grid_stack=grid_stack),
                                           image_plane_grid_stacks))

        planes = []

        for plane_index in range(0, len(ordered_redshifts)):

            if plane_index < len(ordered_redshifts) - 1:
                compute_deflections = True
            elif plane_index == len(ordered_redshifts) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_grid_stacks = image_plane_grid_stacks

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = lensing_util.scaling_factor_between_redshifts_for_cosmology(
                        z1=ordered_redshifts[previous_plane_index], z2=ordered_redshifts[plane_index],
                        z_final=ordered_redshifts[-1], cosmology=cosmology)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    if planes[previous_plane_index].deflection_stacks is not None:
                        scaled_deflections = list(map(lambda deflection_stack : deflection_stack.apply_function(scale),
                                                      planes[previous_plane_index].deflection_stacks))
                    else:
                        scaled_deflections = None

                    if scaled_deflections is not None:

                        def minus(grid, deflections):
                            return grid - deflections

                        new_grid_stacks = list(map(lambda grid, deflections: grid.map_function(minus, deflections),
                                 new_grid_stacks, scaled_deflections))

            planes.append(pl.PlaneStack(galaxies=galaxies_in_redshift_ordered_lists[plane_index],
                                        grid_stacks=new_grid_stacks, borders=borders,
                                        compute_deflections=compute_deflections, cosmology=cosmology))

        super(TracerMultiPlanesStack, self).__init__(planes=planes, cosmology=cosmology)


class TracerImageSourcePlanesPositions(AbstractTracer):

    def __init__(self, lens_galaxies, image_plane_positions, cosmology=cosmo.Planck15):
        """Positional ray-tracer for a lensing system with two planes, an image-plane and source-plane (source-plane \
        galaxies are not input for the positional ray-tracer, as it is only the proximity that image_plane_positions trace to \
        within one another that needs to be computed).

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. If a cosmology is supplied, the plane's angular diameter distances, \
        conversion factors, etc. are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        image_plane_positions : [[[]]]
            The (y,x) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same location(s) \
            in the source-plane.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        image_plane = pl.PlanePositions(galaxies=lens_galaxies, positions=image_plane_positions,
                                        compute_deflections=True, cosmology=cosmology)

        source_plane_positions = image_plane.trace_to_next_plane()

        source_plane = pl.PlanePositions(galaxies=None, positions=source_plane_positions,
                                         compute_deflections=False, cosmology=cosmology)

        super(TracerImageSourcePlanesPositions, self).__init__(planes=[image_plane, source_plane], cosmology=cosmology)


class TracerMultiPlanesPositions(AbstractTracer):

    def __init__(self, galaxies, image_plane_positions, cosmology=cosmo.Planck15):
        """Positional ray-tracer for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lensing-geometry of the multi-plane system.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_positions : [[[]]]
            The (y,x) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same location(s) \
            in the final source-plane.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        ordered_redshifts = lensing_util.ordered_redshifts_from_galaxies(galaxies=galaxies)

        galaxies_in_redshift_ordered_lists = \
            lensing_util.galaxies_in_redshift_ordered_lists_from_galaxies(galaxies=galaxies,
                                                                           ordered_redshifts=ordered_redshifts)

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        planes = []

        for plane_index in range(0, len(ordered_redshifts)):

            if plane_index < len(ordered_redshifts) - 1:
                compute_deflections = True
            elif plane_index == len(ordered_redshifts) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_positions = image_plane_positions

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = lensing_util.scaling_factor_between_redshifts_for_cosmology(
                        z1=ordered_redshifts[previous_plane_index], z2=ordered_redshifts[plane_index],
                        z_final=ordered_redshifts[-1], cosmology=cosmology)

                    scaled_deflections = list(map(lambda deflections:
                                                  np.multiply(scaling_factor, deflections),
                                                  planes[previous_plane_index].deflections))

                    new_positions = list(map(lambda positions, deflections:
                                             np.subtract(positions, deflections), new_positions, scaled_deflections))

            planes.append(pl.PlanePositions(galaxies=galaxies_in_redshift_ordered_lists[plane_index],
                                                 positions=new_positions, compute_deflections=compute_deflections))

        super(TracerMultiPlanesPositions, self).__init__(planes=planes, cosmology=cosmology)
