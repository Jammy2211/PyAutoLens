from functools import wraps
import math

import numpy as np
from astropy import cosmology as cosmo

from autolens import exc
from autolens.lens.util import lens_util
from autolens.lens import ray_tracing
from autolens.lens.stack import plane_stack
from autolens.model.inversion import pixelizations as pix


class AbstractTracerStack(ray_tracing.AbstractTracer):

    def __init__(self, planes, cosmology):
        """Abstract ray-tracer for lens systems with any number of planes and multiple grid-stacks.

        From the galaxies in the tracer's planes, the image-plane images of each grid-stack are computed.

        Parameters
        ----------
        planes : [plane_stack.Plane]
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
        return list(map(lambda image_plane_image_1d, grid_stack:
                        grid_stack.regular.scaled_array_from_array_1d(image_plane_image_1d),
                        self.image_plane_images_1d, self.image_plane.grid_stacks))

    @property
    def image_plane_images_for_simulation(self):
        return [sum(image_plane_image_of_plane_for_simulation[i] for image_plane_image_of_plane_for_simulation in
                    self.image_plane_images_of_planes_for_simulation) for i in range(self.total_grid_stacks)]

    @property
    def image_plane_images_of_planes_for_simulation(self):
        return [plane.image_plane_images_for_simulation for plane in self.planes]

    @property
    def image_plane_images_1d(self):
        return list(map(lambda image_plane_image_1d_of_planes: sum(image_plane_image_1d_of_planes),
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
        return list(map(lambda image_plane_blurring_image_1d_of_planes: sum(image_plane_blurring_image_1d_of_planes),
                        self.image_plane_blurring_images_1d_of_planes))

    @property
    def image_plane_blurring_images_1d_of_planes(self):
        image_plane_blurring_images_1d = [plane.image_plane_blurring_images_1d for plane in self.planes]
        image_plane_blurring_images_1d_of_planes = [[] for _ in range(self.total_grid_stacks)]
        for image_index in range(self.total_grid_stacks):
            for plane_index in range(self.total_planes):
                image_plane_blurring_images_1d_of_planes[image_index].append(
                    image_plane_blurring_images_1d[plane_index][image_index])
        return image_plane_blurring_images_1d_of_planes


class TracerImagePlaneStack(AbstractTracerStack):

    def __init__(self, lens_galaxies, image_plane_grid_stacks, borders=None, cosmology=cosmo.Planck15):
        """Ray tracer for a lens system with just an image-plane.

        As there is only 1 plane, there are no ray-tracing calculations. This class is therefore only used for fitting \
        image-plane galaxies with light profiles.

        This tracer has a list of grid-stacks (see grid_stack.GridStack) which are all used for ray-tracing.

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

        image_plane = plane_stack.PlaneStack(galaxies=lens_galaxies, grid_stacks=image_plane_grid_stacks, borders=borders,
                                    compute_deflections=True, cosmology=cosmology)

        super(TracerImagePlaneStack, self).__init__(planes=[image_plane], cosmology=cosmology)


class TracerImageSourcePlanesStack(AbstractTracerStack):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grid_stacks, borders=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lens system with two planes, an image-plane and source-plane.

        This tracer has a list of grid-stacks (see grid_stack.GridStack) which are all used for ray-tracing.

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

        image_plane_grid_stacks = list(map(lambda data_grids:
                                           pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
                                               galaxies=source_galaxies,
                                               grid_stack=data_grids),
                                           image_plane_grid_stacks))

        image_plane = plane_stack.PlaneStack(galaxies=lens_galaxies, grid_stacks=image_plane_grid_stacks, borders=borders,
                                    compute_deflections=True)

        source_plane_grid_stacks = image_plane.trace_grids_to_next_plane()

        source_plane = plane_stack.PlaneStack(galaxies=source_galaxies, grid_stacks=source_plane_grid_stacks,
                                     borders=borders, compute_deflections=False, cosmology=cosmology)

        super(TracerImageSourcePlanesStack, self).__init__(planes=[image_plane, source_plane], cosmology=cosmology)


class TracerMultiPlanesStack(AbstractTracerStack):

    def __init__(self, galaxies, image_plane_grid_stacks, borders=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lens system with any number of planes.

        To perform multi-plane ray-tracing, the cosmology that is input is used to rescale deflection-angles \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has a list of grid-stacks (see grid_stack.GridStack) which are all used for ray-tracing.

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

        ordered_redshifts = lens_util.ordered_plane_redshifts_from_galaxies(galaxies=galaxies)

        galaxies_in_redshift_ordered_lists = \
            lens_util.galaxies_in_redshift_ordered_planes_from_galaxies(galaxies=galaxies,
                                                                               plane_redshifts=ordered_redshifts)

        image_plane_grid_stacks = list(map(lambda grid_stack:
                                           pix.setup_image_plane_pixelization_grid_from_galaxies_and_grid_stack(
                                               galaxies=galaxies,
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

                    scaling_factor = lens_util.scaling_factor_between_redshifts_for_cosmology(
                        z1=ordered_redshifts[previous_plane_index], z2=ordered_redshifts[plane_index],
                        z_final=ordered_redshifts[-1], cosmology=cosmology)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    if planes[previous_plane_index].deflection_stacks is not None:
                        scaled_deflections = list(map(lambda deflection_stack: deflection_stack.apply_function(scale),
                                                      planes[previous_plane_index].deflection_stacks))
                    else:
                        scaled_deflections = None

                    if scaled_deflections is not None:
                        def minus(grid, deflections):
                            return grid - deflections

                        new_grid_stacks = list(map(lambda grid, deflections: grid.map_function(minus, deflections),
                                                   new_grid_stacks, scaled_deflections))

            planes.append(plane_stack.PlaneStack(galaxies=galaxies_in_redshift_ordered_lists[plane_index],
                                        grid_stacks=new_grid_stacks, borders=borders,
                                        compute_deflections=compute_deflections, cosmology=cosmology))

        super(TracerMultiPlanesStack, self).__init__(planes=planes, cosmology=cosmology)