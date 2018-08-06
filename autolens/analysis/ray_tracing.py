from autolens import exc

from astropy import constants
import math
import numpy as np


class AbstractTracer(object):
    @property
    def all_planes(self):
        raise NotImplementedError()

    @property
    def galaxy_images(self):
        """
        Returns
        -------
        galaxy_images: [ndarray]
            An image for each galaxy in this ray tracer
        """
        return [galaxy_image for plane in self.all_planes for galaxy_image in plane.galaxy_images]

    @property
    def hyper_galaxies(self):
        return [hyper_galaxy for plane in self.all_planes for hyper_galaxy in
                plane.hyper_galaxies]

    @property
    def galaxies(self):
        return [galaxy for plane in self.all_planes for galaxy in plane.galaxies]

    @property
    def all_with_hyper_galaxies(self):
        return len(list(filter(None, self.hyper_galaxies))) == len(self.galaxies)


class Tracer(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grids, cosmology=None):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        If a cosmology is supplied, the plane's angular diameter distances, conversion factors, etc. will be computed.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        image_plane_grids : mask.GridCollection
            The image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        if cosmology is not None:
            self.geometry = TracerGeometry(redshifts=[lens_galaxies[0].redshift, source_galaxies[0].redshift],
                                           cosmology=cosmology)
        else:
            self.geometry = None

        self.image_plane = Plane(lens_galaxies, image_plane_grids, compute_deflections=True)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = Plane(source_galaxies, source_plane_grids, compute_deflections=False)

    def generate_image_of_galaxy_light_profiles(self):
        """Generate the image of the galaxies over the entire ray trace."""
        return self.image_plane.generate_image_of_galaxy_light_profiles(
        ) + self.source_plane.generate_image_of_galaxy_light_profiles()

    def generate_blurring_image_of_galaxy_light_profiles(self):
        """Generate the image of all galaxy light profiles in the blurring regions of the image."""
        return self.image_plane.blurring_image_from_galaxy_light_profiles(
        ) + self.source_plane.blurring_image_from_galaxy_light_profiles()

    def reconstructors_from_source_plane(self, borders, cluster_mask):
        return self.source_plane.reconstructor_from_plane(borders, cluster_mask)


class MultiTracer(AbstractTracer):

    @property
    def all_planes(self):
        return self.planes

    def __init__(self, galaxies, image_plane_grids, cosmology):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grids : mask.GridCollection
            The image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

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

                    scaled_deflections = self.planes[previous_plane_index].deflections. \
                        apply_function(scale)

                    def subtract_scaled_deflections(grid, scaled_deflection):
                        return np.subtract(grid, scaled_deflection)

                    new_grid = new_grid.map_function(subtract_scaled_deflections, scaled_deflections)

            self.planes.append(Plane(galaxies=self.planes_galaxies[plane_index], grids=new_grid,
                                     compute_deflections=compute_deflections))

    def generate_image_of_galaxy_light_profiles(self):
        """Generate the image of the galaxies over the entire ray trace."""
        return sum(np.array(list(map(lambda plane: plane.generate_image_of_galaxy_light_profiles(),
                                     self.planes))))

    def generate_blurring_image_of_galaxy_light_profiles(self):
        """Generate the image of all galaxy light profiles in the blurring regions of the image."""
        return np.ndarray.sum(np.array(list(map(lambda plane: plane.blurring_image_from_galaxy_light_profiles(),
                                                self.planes))))

    def reconstructors_from_planes(self, borders, cluster_mask):
        return list(map(lambda plane: plane.reconstructor_from_plane(borders, cluster_mask), self.planes))


class TracerGeometry(object):

    def __init__(self, redshifts, cosmology):
        """The geometry of a ray-tracing grid comprising an image-plane and source-plane.

        This sets up the angular diameter distances between each plane and the Earth, and between one another. \
        The critical density of the lens plane is also computed.

        Parameters
        ----------
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


class Plane(object):

    def __init__(self, galaxies, grids, compute_deflections=True):
        """

        Represents a plane, which is a set of galaxies and grids at a given redshift in the lens ray-tracing
        calculation.

        The image-plane coordinates are defined on the observed image's uniform regular grid_coords.
        Calculating its model images from its light profiles exploits this uniformity to perform more efficient and
        precise calculations via an iterative sub-griding approach.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to
        the ImagePlane. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid_coords is preferred and
        won't lead noticeable inaccuracy. For example, computing the light profile of the main lens galaxy, ignoring
        minor lensing effects due to a low mass foreground substructure.

        2) When evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the
        point-source images of a lensed quasar, effects like micro-lensing mean lens-plane modeling will be inaccurate.


        Parameters ---------- galaxies : [Galaxy] The galaxies in the plane. grids :
        mask.GridCollection The grids of (x,y) coordinates in the plane, including the image grid_coords,
        sub-grid_coords, blurring, grid_coords, etc.
        """
        self.galaxies = galaxies
        self.grids = grids

        if compute_deflections:
            def calculate_deflections(grid):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

            self.deflections = self.grids.apply_function(calculate_deflections)

    def trace_to_next_plane(self):
        """Trace the grids to the next plane.

        NOTE : This does not work for multi-plane lensing, which requires one to use the previous plane's deflection
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list.
        """
        return self.grids.map_function(np.subtract, self.deflections)

    def generate_image_of_galaxy_light_profiles(self):
        """Generate the image of the galaxies in this plane."""
        if len(self.galaxies) == 0:
            return np.zeros(self.grids.image.shape[0])
        return intensities_via_sub_grid(self.grids.sub, self.galaxies)

    @property
    def galaxy_images(self):
        """
        Returns
        -------
        galaxy_images: [ndarray]
            A list of images of galaxies in this plane
        """
        return [self.image_from_galaxy(galaxy) for galaxy in self.galaxies]

    @property
    def hyper_galaxies(self):
        return [galaxy.hyper_galaxy for galaxy in self.galaxies]

    def image_from_galaxy(self, galaxy):
        """
        Parameters
        ----------
        galaxy: Galaxy
            An individual galaxy, assumed to be in this plane

        Returns
        -------
        galaxy_image: ndarray
            An array describing the intensity of light coming from the galaxy embedded in this plane
        """
        return intensities_via_sub_grid(self.grids.sub, [galaxy])

    def blurring_image_from_galaxy_light_profiles(self):
        """Generate the image of the galaxies in this plane."""
        return intensities_via_grid(self.grids.blurring, self.galaxies)

    def reconstructor_from_plane(self, borders, sparse_mask):

        pixelized_galaxies = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(pixelized_galaxies) == 0:
            return None
        if len(pixelized_galaxies) == 1:
            return pixelized_galaxies[0].pixelization.reconstructor_from_pix_grids(self.grids, borders, sparse_mask)
        elif len(pixelized_galaxies) > 1:
            raise exc.PixelizationException('The number of galaxies with pixelizations in one plane is above 1')


def intensities_via_sub_grid(sub_grid, galaxies):
    sub_intensities = sum(map(lambda g: g.intensity_from_grid(sub_grid), galaxies))
    return sub_grid.sub_data_to_image(sub_intensities)


def intensities_via_grid(image_grid, galaxies):
    return sum(map(lambda g: g.intensity_from_grid(image_grid), galaxies))


def deflections_for_image_grid(image_grid, galaxies):
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(image_grid), galaxies))


def deflections_for_grids(grids, galaxies):
    return grids.apply_function(lambda grid: deflections_for_image_grid(grid, galaxies))


def traced_collection_for_deflections(grids, deflections):
    def subtract_scaled_deflections(grid, scaled_deflection):
        return np.subtract(grid, scaled_deflection)

    result = grids.map_function(subtract_scaled_deflections, deflections)

    return result
