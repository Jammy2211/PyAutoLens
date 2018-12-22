from functools import wraps

import numpy as np
from astropy import constants
from astropy import cosmology as cosmo

from autolens import exc
from autolens.data.array import scaled_array
from autolens.data.array import grids
from autolens.lensing.util import plane_util


def check_plane_for_redshift(func):
    """If a plane's galaxies do not have redshifts, its cosmological quantities cannot be computed. This wrapper \
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

        if self.redshift is not None:
            return func(self)
        else:
            return None

    return wrapper


class AbstractPlane(object):

    def __init__(self, galaxies, cosmology):
        """An abstract plane which represents a set of galaxies at a given redshift.

        From a plane, the surface-density, potential and deflection angles of the galaxies can be computed, as well as \
        cosmological quantities like angular diameter distances..

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        cosmology : astropy.cosmology
            The cosmology associated with the plane, used to convert arc-second coordinates to physical values.
        """

        self.galaxies = galaxies

        if not galaxies:
            raise exc.RayTracingException('An empty list of galaxies was supplied to Plane')

        if any([redshift is not None for redshift in self.galaxy_redshifts]):
            if not all([galaxies[0].redshift == galaxy.redshift for galaxy in galaxies]):
                raise exc.RayTracingException('The galaxies supplied to A Plane have different redshifts or one galaxy '
                                              'does not have a redshift.')

        self.cosmology = cosmology

    @property
    def primary_grid_stack(self):
        return NotImplementedError()

    @property
    def galaxy_redshifts(self):
        return [galaxy.redshift for galaxy in self.galaxies]

    @property
    def redshift(self):
        return self.galaxies[0].redshift

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    @property
    @check_plane_for_redshift
    def arcsec_per_kpc_proper(self):
        return self.cosmology.arcsec_per_kpc_proper(z=self.redshift).value

    @property
    @check_plane_for_redshift
    def kpc_per_arcsec_proper(self):
        return 1.0 / self.arcsec_per_kpc_proper

    @property
    @check_plane_for_redshift
    def angular_diameter_distance_to_earth(self):
        return self.cosmology.angular_diameter_distance(self.redshift).to('kpc').value

    @property
    def has_light_profile(self):
        return any(list(map(lambda galaxy: galaxy.has_light_profile, self.galaxies)))

    @property
    def has_pixelization(self):
        return any(list(map(lambda galaxy: galaxy.has_pixelization, self.galaxies)))

    @property
    def has_regularization(self):
        return any(list(map(lambda galaxy: galaxy.has_regularization, self.galaxies)))

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda galaxy: galaxy.has_hyper_galaxy, self.galaxies)))

    @property
    def hyper_galaxies(self):
        return [galaxy.hyper_galaxy for galaxy in self.galaxies]

    @property
    def surface_density(self):
        surface_density_1d = plane_util.surface_density_of_galaxies_from_grid(grid=self.primary_grid_stack.sub.unlensed_grid,
                                                                              galaxies=self.galaxies)
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(surface_density_1d)

    @property
    def potential(self):
        potential_1d = plane_util.potential_of_galaxies_from_grid(grid=self.primary_grid_stack.sub.unlensed_grid,
                                                                  galaxies=self.galaxies)
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(potential_1d)

    @property
    def deflections_y(self):
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(self.deflections_1d[:, 0])

    @property
    def deflections_x(self):
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(self.deflections_1d[:, 1])

    @property
    def deflections_1d(self):
        return plane_util.deflections_of_galaxies_from_grid(grid=self.primary_grid_stack.sub.unlensed_grid,
                                                            galaxies=self.galaxies)

    def luminosities_of_galaxies_within_circles(self, radius, conversion_factor=1.0):
        """Compute the total luminosity of all galaxies in this plane within a circle of specified radius.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the photometric zeropoint).

        See *galaxy.light_within_circle* and *light_profiles.light_within_circle* for details
        of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless luminosity within.
        conversion_factor : float
            Factor the dimensionless luminosity is multiplied by to convert it to a physical luminosity \ 
            (e.g. a photometric zeropoint).                
        """
        return list(map(lambda galaxy : galaxy.luminosity_within_circle(radius, conversion_factor),
                        self.galaxies))

    def luminosities_of_galaxies_within_ellipses(self, major_axis, conversion_factor=1.0):
        """
        Compute the total luminosity of all galaxies in this plane within a ellipse of specified major-axis.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the photometric zeropoint).

        See *galaxy.light_within_ellipse* and *light_profiles.light_within_ellipse* for details
        of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major-axis of the ellipse to compute the dimensionless luminosity within.
        conversion_factor : float
            Factor the dimensionless luminosity is multiplied by to convert it to a physical luminosity \ 
            (e.g. a photometric zeropoint).            
        """
        return list(map(lambda galaxy : galaxy.luminosity_within_ellipse(major_axis, conversion_factor),
                        self.galaxies))

    def masses_of_galaxies_within_circles(self, radius, conversion_factor=1.0):
        """
        Compute the total mass of all galaxies in this plane within a circle of specified radius.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the critical surface mass density).

        See *galaxy.mass_within_circle* and *mass_profiles.mass_within_circle* for details
        of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        conversion_factor : float
            Factor the dimensionless mass is multiplied by to convert it to a physical mass (e.g. the critical surface \
            mass density).            
        """
        return list(map(lambda galaxy : galaxy.mass_within_circle(radius, conversion_factor),
                        self.galaxies))

    def masses_of_galaxies_within_ellipses(self, major_axis, conversion_factor=1.0):
        """
        Compute the total mass of all galaxies in this plane within a ellipse of specified major-axis.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the critical surface mass density).

        See *galaxy.mass_within_ellipse* and *mass_profiles.mass_within_ellipse* for details
        of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major-axis of the ellipse to compute the dimensionless mass within.
        conversion_factor : float
            Factor the dimensionless mass is multiplied by to convert it to a physical mass (e.g. the critical surface \
            mass density).            
        """
        return list(map(lambda galaxy : galaxy.mass_within_ellipse(major_axis, conversion_factor),
                        self.galaxies))


class Plane(AbstractPlane):

    def __init__(self, galaxies, grid_stack, border=None, compute_deflections=True, cosmology=cosmo.Planck15):
        """A plane which uses one grid-stack of (y,x) grid_stack (e.g. a regular-grid, sub-grid, etc.)

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grid_stack : masks.GridStack
            The stack of grid_stacks of (y,x) arc-second coordinates of this plane.
        border : masks.RegularGridBorder
            The borders of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
            source-plane borders.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        cosmology : astropy.cosmology
            The cosmology associated with the plane, used to convert arc-second coordinates to physical values.
        """

        super(Plane, self).__init__(galaxies=galaxies, cosmology=cosmology)

        self.grid_stack = grid_stack
        self.border = border

        if compute_deflections:

            def calculate_deflections(grid):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

            self.deflection_stack = self.grid_stack.apply_function(calculate_deflections)

        else:
            self.deflection_stack = None
        self.cosmology = cosmology

    def trace_grids_to_next_plane(self):
        """Trace this plane's grid_stacks to the next plane, using its deflection angles."""

        def minus(grid, deflections):
            return grid - deflections

        return self.grid_stack.map_function(minus, self.deflection_stack)

    @property
    def primary_grid_stack(self):
        return self.grid_stack

    @property
    def has_padded_grid_stack(self):
        return isinstance(self.grid_stack.regular, grids.PaddedRegularGrid)

    @property
    def mapper(self):

        galaxies_with_pixelization = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(galaxies_with_pixelization) == 0:
            return None
        if len(galaxies_with_pixelization) == 1:
            pixelization = galaxies_with_pixelization[0].pixelization
            return pixelization.mapper_from_grid_stack_and_border(self.grid_stack, self.border)
        elif len(galaxies_with_pixelization) > 1:
            raise exc.PixelizationException('The number of galaxies with pixelizations in one plane is above 1')

    @property
    def regularization(self):

        galaxies_with_regularization = list(filter(lambda galaxy: galaxy.has_regularization, self.galaxies))

        if len(galaxies_with_regularization) == 0:
            return None
        if len(galaxies_with_regularization) == 1:
            return galaxies_with_regularization[0].regularization
        elif len(galaxies_with_regularization) > 1:
            raise exc.PixelizationException('The number of galaxies with regularizations in one plane is above 1')

    @property
    def image_plane_image(self):
        return self.grid_stack.regular.scaled_array_from_array_1d(self.image_plane_image_1d)

    @property
    def image_plane_image_for_simulation(self):
        if not self.has_padded_grid_stack:
            raise exc.RayTracingException(
                'To retrieve an image plane image for the simulation, the grid_stacks in the tracer_normal'
                'must be padded grid_stacks')
        return self.grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=self.image_plane_image_1d)

    @property
    def image_plane_image_1d(self):
        return plane_util.intensities_of_galaxies_from_grid(grid=self.primary_grid_stack.sub, galaxies=self.galaxies)

    @property
    def image_plane_image_1d_of_galaxies(self):
        return [plane_util.intensities_of_galaxies_from_grid(grid=self.grid_stack.sub, galaxies=[galaxy]) 
                for galaxy in self.galaxies]

    @property
    def image_plane_blurring_image_1d(self):
        return plane_util.intensities_of_galaxies_from_grid(grid=self.primary_grid_stack.blurring, galaxies=self.galaxies)

    @property
    def plane_image(self):
        return plane_util.plane_image_of_galaxies_from_grid(shape=self.grid_stack.regular.mask.shape,
                                                            grid=self.grid_stack.regular,
                                                            galaxies=self.galaxies)

    @property
    def yticks(self):
        """Compute the yticks labels of this grid_stack, used for plotting the y-axis ticks when visualizing an image \
        """
        return np.linspace(np.amin(self.grid_stack.regular[:, 0]), np.amax(self.grid_stack.regular[:, 0]), 4)

    @property
    def xticks(self):
        """Compute the xticks labels of this grid_stack, used for plotting the x-axis ticks when visualizing an \
        image"""
        return np.linspace(np.amin(self.grid_stack.regular[:, 1]), np.amax(self.grid_stack.regular[:, 1]), 4)


class PlanePositions(object):

    def __init__(self, galaxies, positions, compute_deflections=True, cosmology=None):
        """A plane represents a set of galaxies at a given redshift in a ray-tracer_normal and the positions of image-plane \
        coordinates which mappers close to one another in the source-plane.

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        positions : [[[]]]
            The (y,x) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same
            location(s) in the final source-plane.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        """

        self.galaxies = galaxies
        self.positions = positions

        if compute_deflections:
            def calculate_deflections(positions):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(positions), galaxies))

            self.deflections = list(map(lambda positions: calculate_deflections(positions), self.positions))

        self.cosmology = cosmology

    def trace_to_next_plane(self):
        """Trace the positions to the next plane."""
        return list(map(lambda positions, deflections: np.subtract(positions, deflections),
                        self.positions, self.deflections))


class PlaneImage(scaled_array.ScaledRectangularPixelArray):

    def __init__(self, array, pixel_scales, grid, origin=(0.0, 0.0)):
        self.grid = grid
        super(PlaneImage, self).__init__(array=array, pixel_scales=pixel_scales, origin=origin)
