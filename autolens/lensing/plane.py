from functools import wraps

import numpy as np
from astropy import constants

from autolens import exc
from autolens.data.array.util import grid_util
from autolens.data.array.util import mapping_util
from autolens.data.array import grids, scaled_array


def check_plane_cosmology(func):
    """
    Wrap the function in a function that, if the grid_stack is a sub-grid_stack (grid_stacks.SubGrid), rebins the computed values to  the
    image-grid_stack by taking the mean of each set of sub-gridded values.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
        A function that requires transformed coordinates
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
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

        if self.cosmology is not None and self.redshift is not None:
            return func(self, *args, *kwargs)
        else:
            return None

    return wrapper


class AbstractPlane(object):

    def __init__(self, galaxies, cosmology=None):
        """An abstract plane which represents a set of galaxies at a given redshift in a ray-tracer and the grid_stack of \
        image-plane lensed coordinates.

        From a plane, the image's of its galaxies can be computed (in both the image-plane and source-plane). The \
        surface-density, potential and deflection angles of the galaxies can also be computed.

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        borders : masks.RegularGridBorder
            The borders of the regular-grid_stack, which is used to relocate demagnified traced regular-pixel to the \
            source-plane borders.
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
    @check_plane_cosmology
    def arcsec_per_kpc_proper(self):
        return self.cosmology.arcsec_per_kpc_proper(z=self.redshift).value

    @property
    @check_plane_cosmology
    def kpc_per_arcsec_proper(self):
        return 1.0 / self.arcsec_per_kpc_proper

    @property
    @check_plane_cosmology
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
        surface_density_1d = sum([surface_density_from_grid(self.primary_grid_stack.sub.unlensed_grid, [galaxy])
                                  for galaxy in self.galaxies])
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(surface_density_1d)

    @property
    def potential(self):
        potential_1d = sum([potential_from_grid(self.primary_grid_stack.sub.unlensed_grid, [galaxy]) for galaxy in self.galaxies])
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(potential_1d)

    @property
    def deflections_y(self):
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(self.deflections_1d[:, 0])

    @property
    def deflections_x(self):
        return self.primary_grid_stack.regular.scaled_array_from_array_1d(self.deflections_1d[:, 1])

    @property
    def deflections_1d(self):
        return sum([deflections_from_grid(self.primary_grid_stack.sub.unlensed_grid, [galaxy])
                    for galaxy in self.galaxies])

    def luminosities_of_galaxies_within_circles(self, radius, conversion_factor=1.0):
        """
        Compute the total luminosity of all galaxies in this plane within a circle of specified radius.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the photometric zeropoint).

        See *galaxy.light_within_circle* and *light_profiles.light_within_circle* for details
        of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless luminosity within.
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
        """
        return list(map(lambda galaxy : galaxy.mass_within_ellipse(major_axis, conversion_factor),
                        self.galaxies))


class Plane(AbstractPlane):

    def __init__(self, galaxies, grid_stack, border=None, compute_deflections=True, cosmology=None):
        """A plane which uses just one stacked grid_stack of coordinates (e.g. a regular-grid_stack, sub-grid_stack, etc.)

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grid_stack : masks.DataGridStack
            The stack of grid_stacks of (x,y) arc-second coordinates of this plane.
        border : masks.RegularGridBorder
            The borders of the regular-grid_stack, which is used to relocate demagnified traced regular-pixel to the \
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
        return sum([intensities_from_grid(grid=self.grid_stack.sub, galaxies=[galaxy]) for galaxy in self.galaxies])

    @property
    def image_plane_image_1d_of_galaxies(self):
        return [intensities_from_grid(grid=self.grid_stack.sub, galaxies=[galaxy]) for galaxy in self.galaxies]

    @property
    def image_plane_blurring_image_1d(self):
        return sum([intensities_from_grid(grid=self.grid_stack.blurring, galaxies=[galaxy]) for galaxy in self.galaxies])

    @property
    def plane_image(self):
        return plane_image_from_grid_and_galaxies(shape=self.grid_stack.regular.mask.shape,
                                                  grid_stack=self.grid_stack.regular, galaxies=self.galaxies)

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


class PlaneStack(AbstractPlane):

    def __init__(self, galaxies, grid_stacks, borders=None, compute_deflections=True, cosmology=None):
        """A plane which uses just one stacked grid_stack of coordinates (e.g. a regular-grid_stack, sub-grid_stack, etc.)

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grid_stacks : masks.DataGridStack
            The stack of grid_stacks of (x,y) arc-second coordinates of this plane.
        borders : masks.RegularGridBorder
            The borders of the regular-grid_stack, which is used to relocate demagnified traced regular-pixel to the \
            source-plane borders.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        cosmology : astropy.cosmology
            The cosmology associated with the plane, used to convert arc-second coordinates to physical values.
        """

        super(PlaneStack, self).__init__(galaxies=galaxies, cosmology=cosmology)

        self.grid_stacks = grid_stacks
        self.borders = borders

        if compute_deflections:

            def calculate_deflections(grid):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

            self.deflection_stacks = list(map(lambda grid_stack: grid_stack.apply_function(calculate_deflections), self.grid_stacks))

        else:
            self.deflection_stacks = None

    def trace_grids_to_next_plane(self):
        """Trace this plane's grid_stacks to the next plane, using its deflection angles."""

        def minus(grid, deflections):
            return grid - deflections

        return list(
            map(lambda grid, deflections: grid.map_function(minus, deflections), self.grid_stacks, self.deflection_stacks))

    @property
    def primary_grid_stack(self):
        return self.grid_stacks[0]

    @property
    def total_grid_stacks(self):
        return len(self.grid_stacks)

    @property
    def has_padded_grid_stack(self):
        return any(list(map(lambda grid_stack : isinstance(grid_stack.regular, grids.PaddedRegularGrid),
                            self.grid_stacks)))

    @property
    def mapper(self):

        galaxies_with_pixelization = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(galaxies_with_pixelization) == 0:
            return None
        if len(galaxies_with_pixelization) == 1:
            pixelization = galaxies_with_pixelization[0].pixelization
            return pixelization.mapper_from_grid_stack_and_border(self.grid_stacks[0], self.borders)
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
    def image_plane_images(self):
        return list(map(lambda image_plane_image_1d, grid_stack :
                        grid_stack.regular.scaled_array_from_array_1d(array_1d=image_plane_image_1d),
                        self.image_plane_images_1d, self.grid_stacks))

    @property
    def image_plane_images_for_simulation(self):
        if not self.has_padded_grid_stack:
            raise exc.RayTracingException(
                'To retrieve an image plane image for the simulation, the grid_stack in the tracer_normal'
                'must be padded grid_stack')
        return list(map(lambda image_plane_image_1d, grid_stack : 
                        grid_stack.regular.map_to_2d_keep_padded(padded_array_1d=image_plane_image_1d),
                        self.image_plane_images_1d, self.grid_stacks))

    @property
    def image_plane_images_1d(self):
        return list(map(lambda grid_stack :
               sum([intensities_from_grid(grid=grid_stack.sub, galaxies=[galaxy]) for galaxy in self.galaxies]),
                        self.grid_stacks))

    @property
    def image_plane_images_1d_of_galaxies(self):
        return list(map(lambda grid_stack : [intensities_from_grid(grid=grid_stack.sub, galaxies=[galaxy])
                                             for galaxy in self.galaxies],
                        self.grid_stacks))

    @property
    def image_plane_blurring_images_1d(self):
        return list(map(lambda grid_stack :
               sum([intensities_from_grid(grid=grid_stack.blurring, galaxies=[galaxy]) for galaxy in self.galaxies]),
                        self.grid_stacks))

    @property
    def plane_images(self):
        return list(map(lambda grid_stack : plane_image_from_grid_and_galaxies(shape=grid_stack.regular.mask.shape,
                                         grid_stack=grid_stack.regular, galaxies=self.galaxies), self.grid_stacks))
    @property
    def yticks(self):
        """Compute the yticks labels of this grid_stack, used for plotting the y-axis ticks when visualizing an \
         image"""
        return list(map(lambda grid_stack :
                        np.linspace(np.amin(grid_stack.regular[:,0]), np.amax(grid_stack.regular[:,0]), 4),
                        self.grid_stacks))

    @property
    def xticks(self):
        """Compute the xticks labels of this grid_stack, used for plotting the x-axis ticks when visualizing an \
        image"""
        return list(map(lambda grid_stack :
                        np.linspace(np.amin(grid_stack[0].regular[:,1]), np.amax(grid_stack[0].regular[:,1]), 4),
                        self.grid_stacks))


class PlanePositions(object):

    def __init__(self, galaxies, positions, compute_deflections=True, cosmology=None):
        """A plane represents a set of galaxies at a given redshift in a ray-tracer_normal and the positions of image-plane \
        coordinates which mappers close to one another in the source-plane.

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        positions : [[[]]]
            The (x,y) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same
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


def sub_to_image_grid(func):
    """
    Wrap the function in a function that, if the grid_stack is a sub-grid_stack (grid_stacks.SubGrid), rebins the computed values to
    the image-grid_stack by taking the mean of each set of sub-gridded values.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
        A function that requires transformed coordinates
    """

    @wraps(func)
    def wrapper(grid, galaxies, *args, **kwargs):
        """

        Parameters
        ----------
        grid : ndarray
            PlaneCoordinates in either cartesian or profiles coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        result = func(grid, galaxies, *args, *kwargs)

        if isinstance(grid, grids.SubGrid):
            return grid.sub_data_to_regular_data(result)
        else:
            return result

    return wrapper


@sub_to_image_grid
def intensities_from_grid(grid, galaxies):
    return sum(map(lambda g: g.intensities_from_grid(grid), galaxies))


@sub_to_image_grid
def surface_density_from_grid(grid, galaxies):
    return sum(map(lambda g: g.surface_density_from_grid(grid), galaxies))


@sub_to_image_grid
def potential_from_grid(grid, galaxies):
    return sum(map(lambda g: g.potential_from_grid(grid), galaxies))


# TODO : There will be a much cleaner way to apply sub datas to surface_density to the array wihtout the need for a
# transpose

def deflections_from_grid(grid, galaxies):
    deflections = sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))
    if isinstance(grid, grids.SubGrid):
        return np.asarray([grid.sub_data_to_regular_data(deflections[:, 0]),
                           grid.sub_data_to_regular_data(deflections[:, 1])]).T
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))


def deflections_from_sub_grid(sub_grid, galaxies):
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(sub_grid), galaxies))


def deflections_from_grid_stack(grid_stack, galaxies):
    return grid_stack.apply_function(lambda grid: deflections_from_sub_grid(grid, galaxies))


def plane_image_from_grid_and_galaxies(shape, grid_stack, galaxies, buffer=1.0e-2):

    y_min = np.min(grid_stack[:, 0]) - buffer
    y_max = np.max(grid_stack[:, 0]) + buffer
    x_min = np.min(grid_stack[:, 1]) - buffer
    x_max = np.max(grid_stack[:, 1]) + buffer

    pixel_scales = (float((y_max - y_min) / shape[0]), float((x_max - x_min) / shape[1]))
    origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

    uniform_grid = grid_util.regular_grid_1d_masked_from_mask_pixel_scales_and_origin(mask=np.full(shape=shape,
                                                                                                   fill_value=False),
                                                                                      pixel_scales=pixel_scales,
                                                                                      origin=origin)

    image_1d = sum([intensities_from_grid(uniform_grid, [galaxy]) for galaxy in galaxies])

    image_2d = mapping_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=image_1d, shape=shape)

    im = PlaneImage(array=image_2d, pixel_scales=pixel_scales, grid=grid_stack, origin=origin)

    return PlaneImage(array=image_2d, pixel_scales=pixel_scales, grid=grid_stack, origin=origin)


def traced_collection_for_deflections(grid_stack, deflections):
    def subtract_scaled_deflections(grid, scaled_deflection):
        return np.subtract(grid, scaled_deflection)

    result = grid_stack.map_function(subtract_scaled_deflections, deflections)

    return result
