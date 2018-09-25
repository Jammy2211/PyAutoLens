import math
from functools import wraps

import numpy as np
from astropy import constants
from astropy import cosmology as cosmo

from autolens import exc
from autolens.imaging import imaging_util
from autolens.imaging import mask as msk


class TracerGeometry(object):

    def __init__(self, redshifts, cosmology):
        """The geometry of a ray-tracer, comprising an arbritrary number of planes.

        This allows one to compute the angular diameter distances and critical densities between each plane and \
        the Earth.

        Parameters
        ----------
        redshifts : [float]
            The redshifts of the plane's of this tracer.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        self.cosmology = cosmology
        self.redshifts = redshifts
        self.final_plane = len(self.redshifts) - 1
        self.ang_to_final_plane = self.ang_to_earth(plane_i=self.final_plane)

    def arcsec_per_kpc(self, plane_i):
        return self.cosmology.arcsec_per_kpc_proper(z=self.redshifts[plane_i]).value

    def kpc_per_arcsec(self, plane_i):
        return 1.0 / self.cosmology.arcsec_per_kpc_proper(z=self.redshifts[plane_i]).value

    def ang_to_earth(self, plane_i):
        return self.cosmology.angular_diameter_distance(self.redshifts[plane_i]).to('kpc').value

    def ang_between_planes(self, plane_i, plane_j):
        return self.cosmology.angular_diameter_distance_z1z2(self.redshifts[plane_i], self.redshifts[plane_j]). \
            to('kpc').value

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    def critical_density_kpc(self, plane_i, plane_j):
        return self.constant_kpc * self.ang_to_earth(plane_j) / \
               (self.ang_between_planes(plane_i, plane_j) * self.ang_to_earth(plane_i))

    def critical_density_arcsec(self, plane_i, plane_j):
        return self.critical_density_kpc(plane_i, plane_j) * self.kpc_per_arcsec(plane_i) ** 2.0

    def scaling_factor(self, plane_i, plane_j):
        return (self.ang_between_planes(plane_i, plane_j) * self.ang_to_final_plane) / (
                self.ang_to_earth(plane_j) * self.ang_between_planes(plane_i, self.final_plane))


class AbstractTracer(object):
    image_plane_grids = None

    @property
    def all_planes(self):
        raise NotImplementedError()

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
    def has_unmasked_grids(self):
        return isinstance(self.all_planes[0].grids.image, msk.ImageUnmaskedGrid)

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda galaxy: galaxy.has_hyper_galaxy, self.galaxies)))

    @property
    def galaxies(self):
        return [galaxy for plane in self.all_planes for galaxy in plane.galaxies]

    @property
    def hyper_galaxies(self):
        return [hyper_galaxy for plane in self.all_planes for hyper_galaxy in
                plane.hyper_galaxies]

    @property
    def all_with_hyper_galaxies(self):
        return len(list(filter(None, self.hyper_galaxies))) == len(self.galaxies)

    @property
    def image_plane_image(self):
        return sum(self.image_plane_images_of_planes)

    @property
    def image_plane_images_of_planes(self):
        return list(map(lambda image: self.image_plane.grids.image.map_to_2d(image), self._image_plane_images_of_planes))

    @property
    def image_plane_images_of_galaxies(self):
        return list(map(lambda image: self.image_plane.grids.image.map_to_2d(image), self._image_plane_images_of_galaxies))

    @property
    def image_plane_image_for_simulation(self):
        if not self.has_unmasked_grids:
            raise exc.RayTracingException(
                'To retrieve an _image plane _image for the simulation, the grids in the tracer'
                'must be unmasked grids')
        return sum(map(lambda image: self.image_plane.grids.image.map_to_2d_keep_padded(image),
                       self._image_plane_images_of_planes))

    def plane_images_of_planes(self, shape=(30, 30)):
        return [plane.plane_image(shape) for plane in self.all_planes]

    @property
    def image_grids_of_planes(self):
        return [plane.grids.image for plane in self.all_planes]

    @property
    def mappers_of_planes(self):
        return list(filter(None, [plane.mapper for plane in self.all_planes]))

    @property
    def regularization_of_planes(self):
        return list(filter(None, [plane.regularization for plane in self.all_planes]))

    @property
    def xticks_of_planes(self):
        return [plane.xticks_from_image_grid for plane in self.all_planes]

    @property
    def yticks_of_planes(self):
        return [plane.yticks_from_image_grid for plane in self.all_planes]

    @property
    def _image_plane_image(self):
        return sum(self._image_plane_images_of_planes)

    @property
    def _image_plane_images_of_planes(self):
        return [plane._image_plane_image for plane in self.all_planes]

    @property
    def _image_plane_images_of_galaxies(self):
        return [galaxy_image for plane in self.all_planes for galaxy_image in plane._image_plane_images_of_galaxies]

    @property
    def _image_plane_blurring_image(self):
        return sum(self._image_plane_blurring_images_of_planes)

    @property
    def _image_plane_blurring_images_of_planes(self):
        return [plane._image_plane_blurring_image for plane in self.all_planes]

    @property
    def _image_plane_blurring_images_of_galaxies(self):
        return [galaxy_blurring_image for plane in self.all_planes for galaxy_blurring_image
                in plane._image_plane_blurring_images_of_galaxies]


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

        if cosmology is not None:
            self.geometry = TracerGeometry(redshifts=[lens_galaxies[0].redshift], cosmology=cosmology)
        else:
            self.geometry = None

        self.image_plane = Plane(lens_galaxies, image_plane_grids, borders=borders, compute_deflections=True)


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

        self.image_plane = Plane(lens_galaxies, image_plane_grids, borders=borders, compute_deflections=True)

        if not source_galaxies:
            raise exc.RayTracingException(
                'No source galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        if cosmology is not None:
            self.geometry = TracerGeometry(redshifts=[lens_galaxies[0].redshift, source_galaxies[0].redshift],
                                           cosmology=cosmology)
        else:
            self.geometry = None

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = Plane(source_galaxies, source_plane_grids, borders=borders, compute_deflections=False)

    @property
    def surface_density(self):
        return sum(self.surface_density_of_galaxies)

    @property
    def surface_density_of_galaxies(self):
        return list(map(lambda surface_density: self.image_plane.grids.image.map_to_2d(surface_density),
                        self.image_plane._surface_density_of_galaxies))

    @property
    def potential(self):
        return sum(self.potential_of_galaxies)

    @property
    def potential_of_galaxies(self):
        return list(map(lambda potential: self.image_plane.grids.image.map_to_2d(potential),
                        self.image_plane._potential_of_galaxies))
    
    @property
    def deflections_x(self):
        return sum(self.deflections_x_of_galaxies)

    @property
    def deflections_x_of_galaxies(self):
        return list(map(lambda deflections: self.image_plane.grids.image.map_to_2d(deflections[:,0]),
                        self.image_plane._deflections_of_galaxies))

    @property
    def deflections_y(self):
        return sum(self.deflections_y_of_galaxies)

    @property
    def deflections_y_of_galaxies(self):
        return list(map(lambda deflections: self.image_plane.grids.image.map_to_2d(deflections[:,1]),
                        self.image_plane._deflections_of_galaxies))


class AbstractTracerMulti(AbstractTracer):

    @property
    def all_planes(self):
        return self.planes

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

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerMulti)')

        self.galaxies_redshift_order = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

        # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
        # Using a list of class attributes so make a list of redshifts for now.

        galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, self.galaxies_redshift_order))
        self.planes_redshift_order = [redshift for i, redshift in enumerate(galaxy_redshifts)
                                      if redshift not in galaxy_redshifts[:i]]
        self.geometry = TracerGeometry(redshifts=self.planes_redshift_order, cosmology=cosmology)

        # TODO : Idea is to get a list of all galaxies in each plane - can you clean up the logic below?

        self.planes_galaxies = []

        for (plane_index, plane_redshift) in enumerate(self.planes_redshift_order):
            self.planes_galaxies.append(list(map(lambda galaxy:
                                                 galaxy if galaxy.redshift == plane_redshift else None,
                                                 self.galaxies_redshift_order)))
            self.planes_galaxies[plane_index] = list(filter(None, self.planes_galaxies[plane_index]))

    @property
    def image_plane(self):
        return self.planes[0]


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

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerMulti)')

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
                    scaling_factor = self.geometry.scaling_factor(plane_i=previous_plane_index,
                                                                  plane_j=plane_index)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    if self.planes[previous_plane_index].deflections is not None:
                        scaled_deflections = self.planes[previous_plane_index].deflections.apply_function(scale)
                    else:
                        scaled_deflections = None

                    def subtract_scaled_deflections(grid, scaled_deflection):
                        return np.subtract(grid, scaled_deflection)

                    if scaled_deflections is not None:
                        new_grid = new_grid.map_function(subtract_scaled_deflections, scaled_deflections)
                    else:
                        new_grid = None

            self.planes.append(Plane(galaxies=self.planes_galaxies[plane_index], grids=new_grid, borders=borders,
                                     compute_deflections=compute_deflections))


class Plane(object):

    def __init__(self, galaxies, grids, borders=None, compute_deflections=True):
        """A plane represents a set of galaxies at a given redshift in a ray-tracer and a the grid of _image-plane \
        or lensed coordinates.

        From a plane, the _image's of its galaxies can be computed (in both the _image-plane and source-plane). The \
        surface-density, potential and deflection angles of the galaxies can also be computed.

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grids : mask.ImagingGrids
            The grids of (x,y) arc-second coordinates of this plane.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        """
        self.galaxies = galaxies
        self.grids = grids
        self.borders = borders

        if compute_deflections:

            def calculate_deflections(grid):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

            self.deflections = self.grids.apply_function(calculate_deflections)

        else:

            self.deflections = None

    def trace_to_next_plane(self):
        """Trace this plane's grids to the next plane, using its deflection angles."""
        return self.grids.map_function(np.subtract, self.deflections)

    def plane_image(self, shape=(30, 30)):

        class PlaneImage(np.ndarray):

            def __new__(cls, image, grid):
                plane = np.array(image, dtype='float64').view(cls)
                plane.grid = grid
                return plane

        grid = uniform_grid_from_lensed_grid(self.grids.image, shape)
        image_1d = self.plane_image_from_galaxies(grid)
        image = imaging_util.map_unmasked_1d_array_to_2d_array_from_array_1d_and_shape(array_1d=image_1d, shape=shape)
        return PlaneImage(image=image, grid=self.grids.image)

    def plane_image_from_galaxies(self, plane_grid):
        return sum([intensities_from_grid(plane_grid, [galaxy]) for galaxy in self.galaxies])

    @property
    def hyper_galaxies(self):
        return list(filter(None.__ne__, [galaxy.hyper_galaxy for galaxy in self.galaxies]))

    @property
    def mapper(self):

        galaxies_with_pixelization = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(galaxies_with_pixelization) == 0:
            return None
        if len(galaxies_with_pixelization) == 1:
            pixelization = galaxies_with_pixelization[0].pixelization
            return pixelization.mapper_from_grids_and_borders(self.grids, self.borders)
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
    def xticks_from_image_grid(self):
        return np.around(np.linspace(np.amin(self.grids.image[:, 0]), np.amax(self.grids.image[:, 0]), 4), 2)

    @property
    def yticks_from_image_grid(self):
        return np.around(np.linspace(np.amin(self.grids.image[:, 1]), np.amax(self.grids.image[:, 1]), 4), 2)

    @property
    def _image_plane_image(self):
        return sum(self._image_plane_images_of_galaxies)

    @property
    def _image_plane_images_of_galaxies(self):
        return [self._image_plane_image_from_galaxy(galaxy) for galaxy in self.galaxies]

    def _image_plane_image_from_galaxy(self, galaxy):
        return intensities_from_grid(self.grids.sub, [galaxy])

    @property
    def _image_plane_blurring_image(self):
        return sum(self._image_plane_blurring_images_of_galaxies)

    @property
    def _image_plane_blurring_images_of_galaxies(self):
        return [self._image_plane_blurring_image_from_galaxy(galaxy) for galaxy in self.galaxies]

    def _image_plane_blurring_image_from_galaxy(self, galaxy):
        return intensities_from_grid(self.grids.blurring, [galaxy])

    @property
    def _surface_density(self):
        return sum(self._surface_density_of_galaxies)

    @property
    def _surface_density_of_galaxies(self):
        return [self._surface_density_from_galaxy(galaxy) for galaxy in self.galaxies]

    def _surface_density_from_galaxy(self, galaxy):
        return surface_density_from_grid(self.grids.sub, [galaxy])

    @property
    def _potential(self):
        return sum(self._potential_of_galaxies)

    @property
    def _potential_of_galaxies(self):
        return [self._potential_from_galaxy(galaxy) for galaxy in self.galaxies]

    def _potential_from_galaxy(self, galaxy):
        return potential_from_grid(self.grids.sub, [galaxy])

    @property
    def _deflections(self):
        return sum(self._deflections_of_galaxies)

    @property
    def _deflections_of_galaxies(self):
        return [self._deflections_from_galaxy(galaxy) for galaxy in self.galaxies]

    def _deflections_from_galaxy(self, galaxy):
        return deflections_from_grid(self.grids.sub, [galaxy])
    
    
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

        if cosmology is not None:
            self.geometry = TracerGeometry(redshifts=[lens_galaxies[0].redshift, source_galaxies[0].redshift],
                                           cosmology=cosmology)
        else:
            self.geometry = None

        self.image_plane = PlanePositions(lens_galaxies, positions, compute_deflections=True)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = PlanePositions(None, source_plane_grids, compute_deflections=False)


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
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcePlanes)')

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
                    scaling_factor = self.geometry.scaling_factor(plane_i=previous_plane_index, plane_j=plane_index)
                    scaled_deflections = list(map(lambda deflections:
                                                  np.multiply(scaling_factor, deflections),
                                                  self.planes[previous_plane_index].deflections))

                    new_positions = list(map(lambda positions, deflections:
                                             np.subtract(positions, deflections), new_positions, scaled_deflections))

            self.planes.append(PlanePositions(galaxies=self.planes_galaxies[plane_index], positions=new_positions,
                                              compute_deflections=compute_deflections))


class PlanePositions(object):

    def __init__(self, galaxies, positions, compute_deflections=True):
        """A plane represents a set of galaxies at a given redshift in a ray-tracer and the positions of _image-plane \
        coordinates which mappers close to one another in the source-plane.

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        positions : [[[]]]
            The (x,y) arc-second coordinates of _image-plane pixels which (are expected to) mappers to the same location(s) \
            in the final source-plane.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        """
        self.galaxies = galaxies
        self.positions = positions

        if compute_deflections:
            def calculate_deflections(positions):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(positions), galaxies))

            self.deflections = list(map(lambda positions: calculate_deflections(positions), self.positions))

    def trace_to_next_plane(self):
        """Trace the positions to the next plane."""
        return list(map(lambda positions, deflections: np.subtract(positions, deflections),
                        self.positions, self.deflections))


def sub_to_image_grid(func):
    """
    Wrap the function in a function that, if the grid is a sub-grid (grids.SubGrid), rebins the computed values to \
    the _image-grid by taking the mean of each set of sub-gridded values.

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
        profile : GeometryProfile
            The profiles that owns the function
        grid : ndarray
            PlaneCoordinates in either cartesian or profiles coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        result = func(grid, galaxies, *args, *kwargs)

        if isinstance(grid, msk.SubGrid):
            return grid.sub_data_to_image(result)
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

# TODO : There will be a much cleaner way to apply sub data to surface_density to the array wihtout the need for a transpose

def deflections_from_grid(grid, galaxies):
    deflections = sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))
    if isinstance(grid, msk.SubGrid):
        return np.asarray([grid.sub_data_to_image(deflections[:,0]), grid.sub_data_to_image(deflections[:,1])]).T
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

def deflections_from_sub_grid(sub_grid, galaxies):
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(sub_grid), galaxies))


def deflections_from_grid_collection(grid_collection, galaxies):
    return grid_collection.apply_function(lambda grid: deflections_from_sub_grid(grid, galaxies))


def traced_collection_for_deflections(grids, deflections):
    def subtract_scaled_deflections(grid, scaled_deflection):
        return np.subtract(grid, scaled_deflection)

    result = grids.map_function(subtract_scaled_deflections, deflections)

    return result


def uniform_grid_from_lensed_grid(grid, shape):
    x_min = np.amin(grid[:, 0])
    x_max = np.amax(grid[:, 0])
    y_min = np.amin(grid[:, 1])
    y_max = np.amax(grid[:, 1])

    x_pixel_scale = ((x_max - x_min) / shape[0])
    y_pixel_scale = ((y_max - y_min) / shape[1])

    x_grid = np.linspace(x_min + (x_pixel_scale / 2.0), x_max - (x_pixel_scale / 2.0), shape[0])
    y_grid = np.linspace(y_min + (y_pixel_scale / 2.0), y_max - (y_pixel_scale / 2.0), shape[1])

    source_plane_grid = np.zeros((shape[0] * shape[1], 2))

    i = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            source_plane_grid[i] = np.array([x_grid[x], y_grid[y]])
            i += 1

    return source_plane_grid
