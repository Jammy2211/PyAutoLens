import math

import numpy as np
from astropy import constants
from astropy import cosmology as cosmo

from autolens import exc
from autolens.lensing import plane
from autolens.imaging import mask as msk


class AbstractTracer(object):

    image_plane_grids = None

    @property
    def all_planes(self):
        raise NotImplementedError()

    @property
    def total_planes(self):
        return len(self.all_planes)

    @property
    def redshifts(self):
        return [plane.redshift for plane in self.all_planes]

    @property
    def has_light_profile(self):
        return any(list(map(lambda plane : plane.has_light_profile, self.all_planes)))

    @property
    def has_pixelization(self):
        return any(list(map(lambda plane : plane.has_pixelization, self.all_planes)))

    @property
    def has_regularization(self):
        return any(list(map(lambda plane : plane.has_regularization, self.all_planes)))

    @property
    def has_padded_grids(self):
        return isinstance(self.all_planes[0].grids.image, msk.PaddedImageGrid)

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda plane : plane.has_hyper_galaxy, self.all_planes)))

    @property
    def hyper_galaxies(self):
        return list(filter(None, [hyper_galaxy for plane in self.all_planes for hyper_galaxy in plane.hyper_galaxies]))

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    @property
    def scaled_array_from_array_1d(self):
        return self.image_plane.grids.image.scaled_array_from_array_1d

    @property
    def image_plane_image(self):
        return self.scaled_array_from_array_1d(self._image_plane_image)

    @property
    def image_plane_image_for_simulation(self):
        return sum([plane.image_plane_image_for_simulation for plane in self.all_planes])

    @property
    def mappers_of_planes(self):
        return list(filter(None, [plane.mapper for plane in self.all_planes]))

    @property
    def regularizations_of_planes(self):
        return list(filter(None, [plane.regularization for plane in self.all_planes]))

    @property
    def _image_plane_image(self):
        return sum([plane._image_plane_image for plane in self.all_planes])

    @property
    def _image_plane_blurring_image(self):
        return sum([plane._image_plane_blurring_image for plane in self.all_planes])

    # @property
    # def _image_plane_blurring_images_of_planes(self):
    #     return [plane._image_plane_blurring_image for plane in self.all_planes]
    #
    # @property
    # def _image_plane_blurring_images_of_galaxies(self):
    #     return [galaxy_blurring_image for plane in self.all_planes for galaxy_blurring_image
    #             in plane._image_plane_blurring_images_of_galaxies]

    @property
    def surface_density(self):
        return sum([plane.surface_density for plane in self.all_planes])

    @property
    def potential(self):
        return sum([plane.potential for plane in self.all_planes])

    @property
    def deflections_y(self):
        return sum([plane.deflections_y for plane in self.all_planes])

    @property
    def deflections_x(self):
        return sum([plane.deflections_x for plane in self.all_planes])


class TracerImagePlane(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane]

    def __init__(self, lens_galaxies, image_plane_grids, borders=None, cosmology=None):
        """Ray-tracer for a lensing system with just one plane, the _image-plane. Because there is 1 plane, there are \
        no ray-tracing calculations and the class is used purely for fitting _image-plane galaxies with light \
        profiles.
        
        By default, this has no associated cosmology and galaxy quantities (e.g. effective radii) are in \
        arc-seconds. If a cosmology is supplied, the plane's angular diameter distances, conversion factors, etc. \
        are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the _image-plane.
        image_plane_grids : mask.ImagingGrids
            The _image-plane grids where tracer calculation are performed, (this includes the _image-grid, sub-grid, \
            blurring-grid, etc.).
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        if not lens_galaxies:
            raise exc.RayTracingException('No lens galaxies have been input into the Tracer')

        self.image_plane = plane.Plane(lens_galaxies, image_plane_grids, borders=borders, compute_deflections=True,
                                       cosmology=cosmology)


class TracerImageSourcePlanes(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grids, borders=None, cosmology=None):
        """Ray-tracer for a lensing system with two planes, an _image-plane and source-plane.

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. If a cosmology is supplied, the plane's angular diameter distances, \ 
        conversion factors, etc. are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the _image-plane.
        source_galaxies : [Galaxy]
            The list of galaxies in the source-plane.
        image_plane_grids : mask.ImagingGrids
            The _image-plane grids where ray-tracing calculation are performed, (this includes the _image-grid, \
            sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        self.cosmology = cosmology
        self.image_plane = plane.Plane(lens_galaxies, image_plane_grids, borders=borders, compute_deflections=True,
                                       cosmology=cosmology)

        source_plane_grids = self.image_plane.trace_grids_to_next_plane()

        self.source_plane = plane.Plane(source_galaxies, source_plane_grids, borders=borders, compute_deflections=False,
                                        cosmology=cosmology)

    @property
    @plane.cosmology_check
    def angular_diameter_distance_from_image_to_source_plane(self):
        return self.cosmology.angular_diameter_distance_z1z2(self.image_plane.redshift,
                                                             self.source_plane.redshift).to('kpc').value

    @property
    @plane.cosmology_check
    def critical_density_kpc(self):
        return self.constant_kpc * self.source_plane.angular_diameter_distance_to_earth / \
               (self.angular_diameter_distance_from_image_to_source_plane *
                self.image_plane.angular_diameter_distance_to_earth)

    @property
    @plane.cosmology_check
    def critical_density_arcsec(self):
        return self.critical_density_kpc * self.image_plane.kpc_per_arcsec_proper ** 2.0

    @property
    def source_plane_image(self):
        return self.plane_images_of_planes(shape=(50, 50))


class AbstractTracerMulti(AbstractTracer):

    def __init__(self, galaxies, cosmology):
        """The ray-tracing calculations, defined by a lensing system with just one _image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        imaging_grids : mask.ImagingGrids
            The _image-plane grids where ray-tracing calculation are performed, (this includes the
            _image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        self.cosmology = cosmology

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerMulti)')

        self.galaxies_redshift_order = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

        # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
        # Using a list of class attributes so make a list of redshifts for now.

        galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, self.galaxies_redshift_order))
        self.planes_redshift_order = [redshift for i, redshift in enumerate(galaxy_redshifts)
                                      if redshift not in galaxy_redshifts[:i]]

        # TODO : Idea is to get a list of all galaxies in each plane - can you clean up the logic below?

        self.planes_galaxies = []

        for (plane_index, plane_redshift) in enumerate(self.planes_redshift_order):
            self.planes_galaxies.append(list(map(lambda galaxy:
                                                 galaxy if galaxy.redshift == plane_redshift else None,
                                                 self.galaxies_redshift_order)))
            self.planes_galaxies[plane_index] = list(filter(None, self.planes_galaxies[plane_index]))

    @property
    def all_planes(self):
        return [plane for plane in self.planes]

    @property
    def image_plane(self):
        return self.planes[0]

    @property
    def source_plane_index(self):
        return len(self.planes_redshift_order) - 1

    @property
    def angular_diameter_distance_to_source_plane(self):
        return self.cosmology.angular_diameter_distance(self.planes_redshift_order[-1]).to('kpc').value

    def arcsec_per_kpc_proper_of_plane(self, i):
        return self.cosmology.arcsec_per_kpc_proper(z=self.planes_redshift_order[i]).value

    def kpc_per_arcsec_proper_of_plane(self, i):
        return 1.0 / self.arcsec_per_kpc_proper_of_plane(i)

    def angular_diameter_distance_of_plane_to_earth(self, i):
        return self.cosmology.angular_diameter_distance(self.planes_redshift_order[i]).to('kpc').value

    def angular_diameter_distance_between_planes(self, i, j):
        return self.cosmology.angular_diameter_distance_z1z2(self.planes_redshift_order[i],
                                                             self.planes_redshift_order[j]). \
            to('kpc').value

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    def critical_density_kpc_between_planes(self, i, j):
        return self.constant_kpc * self.angular_diameter_distance_of_plane_to_earth(j) / \
               (self.angular_diameter_distance_between_planes(i, j) * self.angular_diameter_distance_of_plane_to_earth(i))

    def critical_density_arcsec_between_planes(self, i, j):
        return self.critical_density_kpc_between_planes(i, j) * self.kpc_per_arcsec_proper_of_plane(i) ** 2.0

    def scaling_factor_between_planes(self, i, j):
        return (self.angular_diameter_distance_between_planes(i, j) *
                self.angular_diameter_distance_to_source_plane) / \
               (self.angular_diameter_distance_of_plane_to_earth(j) *
                self.angular_diameter_distance_between_planes(i, self.source_plane_index))


class TracerMulti(AbstractTracerMulti):

    def __init__(self, galaxies, image_plane_grids, borders=None, cosmology=cosmo.Planck15):
        """Ray-tracer for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lensing-geometry of the multi-plane system.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grids : mask.ImagingGrids
            The _image-plane grids where ray-tracing calculation are performed, (this includes the
            _image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        super(TracerMulti, self).__init__(galaxies, cosmology)

        self.planes = []

        for plane_index in range(0, len(self.planes_redshift_order)):

            if plane_index < len(self.planes_redshift_order) - 1:
                compute_deflections = True
            elif plane_index == len(self.planes_redshift_order) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_grid = image_plane_grids

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = self.scaling_factor_between_planes(i=previous_plane_index, j=plane_index)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    if self.planes[previous_plane_index].deflections is not None:
                        scaled_deflections = self.planes[previous_plane_index].deflections.apply_function(scale)
                    else:
                        scaled_deflections = None

                    def subtract_scaled_deflections(grid, scaled_deflection):
                        return np.subtract(grid, scaled_deflection)

                    if scaled_deflections is not None:

                        if isinstance(new_grid.image, msk.PaddedImageGrid):
                            image_grid = msk.PaddedImageGrid(arr=new_grid.image - scaled_deflections.image,
                                                             mask=new_grid.image.mask,
                                                             image_shape=new_grid.image.image_shape)
                        elif isinstance(new_grid.image, msk.ImageGrid):
                            image_grid = msk.ImageGrid(arr=new_grid.image - scaled_deflections.image,
                                                       mask=new_grid.image.mask)

                        if isinstance(new_grid.sub, msk.PaddedSubGrid):
                            sub_grid = msk.PaddedSubGrid(new_grid.sub - scaled_deflections.sub, new_grid.sub.mask,
                                                         new_grid.sub.image_shape, new_grid.sub.sub_grid_size)
                        elif isinstance(new_grid.sub, msk.SubGrid):
                            sub_grid = msk.SubGrid(new_grid.sub - scaled_deflections.sub, new_grid.sub.mask,
                                                   new_grid.sub.sub_grid_size)

                        blurring_grid = msk.ImageGrid(arr=new_grid.blurring - scaled_deflections.blurring,
                                                      mask=None)
                        new_grid = msk.ImagingGrids(image=image_grid, sub=sub_grid, blurring=blurring_grid)
                        
                   #     new_grid = new_grid.map_function(subtract_scaled_deflections, scaled_deflections)
                    else:
                        new_grid = None

            self.planes.append(plane.Plane(galaxies=self.planes_galaxies[plane_index], grids=new_grid, borders=borders,
                                     compute_deflections=compute_deflections, cosmology=cosmology))

    
class TracerImageSourcePlanesPositions(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, positions, cosmology=None):
        """Positional ray-tracer for a lensing system with two planes, an _image-plane and source-plane (source-plane \
        galaxies are not input for the positional ray-tracer, as it is only the proximity that positions trace to \
        within one another that needs to be computed).

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. If a cosmology is supplied, the plane's angular diameter distances, \
        conversion factors, etc. are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the _image-plane.
        positions : [[[]]]
            The (x,y) arc-second coordinates of _image-plane pixels which (are expected to) mappers to the same location(s) \
            in the source-plane.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        self.cosmology = cosmology

        self.image_plane = plane.PlanePositions(lens_galaxies, positions, compute_deflections=True,
                                                cosmology=cosmology)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = plane.PlanePositions(None, source_plane_grids, compute_deflections=False,
                                                 cosmology=cosmology)


class TracerMultiPositions(AbstractTracerMulti):

    def __init__(self, galaxies, positions, cosmology):
        """Positional ray-tracer for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lensing-geometry of the multi-plane system.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        positions : [[[]]]
            The (x,y) arc-second coordinates of _image-plane pixels which (are expected to) mappers to the same location(s) \
            in the final source-plane.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourceplane.Planes)')

        super(TracerMultiPositions, self).__init__(galaxies, cosmology)

        self.planes = []

        for plane_index in range(0, len(self.planes_redshift_order)):

            if plane_index < len(self.planes_redshift_order) - 1:
                compute_deflections = True
            elif plane_index == len(self.planes_redshift_order) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_positions = positions

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = self.scaling_factor_between_planes(i=previous_plane_index,
                                                                        j=plane_index)

                    scaled_deflections = list(map(lambda deflections:
                                                  np.multiply(scaling_factor, deflections),
                                                  self.planes[previous_plane_index].deflections))

                    new_positions = list(map(lambda positions, deflections:
                                             np.subtract(positions, deflections), new_positions, scaled_deflections))

            self.planes.append(plane.PlanePositions(galaxies=self.planes_galaxies[plane_index], positions=new_positions,
                                              compute_deflections=compute_deflections))