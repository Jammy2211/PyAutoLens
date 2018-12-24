from functools import wraps

import numpy as np

from autolens import exc

from autolens.data.array import grids
from autolens.lens.util import plane_util
from autolens.lens import plane as pl

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


class PlaneStack(pl.AbstractPlane):

    def __init__(self, galaxies, grid_stacks, borders=None, compute_deflections=True, cosmology=None):
        """A plane which uses a list of grid-stacks of (y,x) grid_stack (e.g. a regular-grid, sub-grid, etc.)

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grid_stacks : masks.GridStack
            The stack of grid_stacks of (y,x) arc-second coordinates of this plane.
        borders : masks.RegularGridBorder
            The borders of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
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
                        plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.sub, galaxies=self.galaxies),
                        self.grid_stacks))

    @property
    def image_plane_images_1d_of_galaxies(self):
        return list(map(lambda grid_stack :
                        [plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.sub, galaxies=[galaxy])
                         for galaxy in self.galaxies],
                        self.grid_stacks))

    @property
    def image_plane_blurring_images_1d(self):
        return list(map(lambda grid_stack :
                        plane_util.intensities_of_galaxies_from_grid(grid=grid_stack.blurring, galaxies=self.galaxies),
                        self.grid_stacks))

    @property
    def plane_images(self):
        return list(map(lambda grid_stack :
                        plane_util.plane_image_of_galaxies_from_grid(shape=grid_stack.regular.mask.shape,
                                                                     grid=grid_stack.regular, galaxies=self.galaxies),
                        self.grid_stacks))
    
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