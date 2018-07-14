from src import exc

from astropy import constants
import math

class Tracer(object):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grids, cosmology=None):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        image_plane_grids : grids.CoordsCollection
            The image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        class TracerGeometry(object):

            def __init__(self, lens_galaxies, source_galaxies, cosmology):

                self.plane_arcsec_per_kpc = [cosmology.arcsec_per_kpc_proper(z=lens_galaxies[0].redshift),
                                             cosmology.arcsec_per_kpc_proper(z=source_galaxies[0].redshift)]
                self.plane_kpc_per_arcsec = list(map(lambda arcsec_per_kpc : 1.0 / arcsec_per_kpc, self.plane_arcsec_per_kpc))

                self.angs_to_earth_kpc = [cosmology.angular_diameter_distance(z=lens_galaxies[0].redshift).to('kpc'),
                                          cosmology.angular_diameter_distance(z=source_galaxies[0].redshift).to('kpc')]

                self.angs_between_planes_kpc = []
                self.angs_between_planes_kpc.append([0.0, cosmology.angular_diameter_distance_z1z2(
                    lens_galaxies[0].redshift, source_galaxies[0].redshift).to('kpc')])
                self.angs_between_planes_kpc.append([cosmology.angular_diameter_distance_z1z2(
                    source_galaxies[0].redshift, lens_galaxies[0].redshift).to('kpc'), 0.0])

                constant_kpc = (constants.c.to('kpc / s').value) ** 2.0 \
                               / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

                self.critical_density_kpc = constant_kpc * self.angs_to_earth_kpc[1] / \
                                            (self.angs_between_planes_kpc[0][1] * self.angs_to_earth_kpc[0])

                self.critical_density_arcsec = self.critical_density_kpc * self.plane_kpc_per_arcsec[0] ** 2.0

        self.cosmology = cosmology

        if self.cosmology is not None:
            self.geometry = TracerGeometry(lens_galaxies, source_galaxies, cosmology)
        else:
            self.geometry = None

        self.image_plane = Plane(lens_galaxies, image_plane_grids, compute_deflections=True)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = Plane(source_galaxies, source_plane_grids, compute_deflections=False)

    def generate_image_of_galaxy_light_profiles(self, mapping):
        """Generate the image of the galaxies over the entire ray trace."""
        return self.image_plane.generate_image_of_galaxy_light_profiles(mapping
        ) + self.source_plane.generate_image_of_galaxy_light_profiles(mapping)

    def generate_blurring_image_of_galaxy_light_profiles(self):
        """Generate the image of all galaxy light profiles in the blurring regions of the image."""
        return self.image_plane.generate_blurring_image_of_galaxy_light_profiles(
        ) + self.source_plane.generate_blurring_image_of_galaxy_light_profiles()

    def generate_pixelization_matrices_of_source_galaxy(self, mapping):
        return self.source_plane.generate_pixelization_matrices_of_galaxy(mapping)


class MultiTracer(object):

    def __init__(self, galaxies, image_plane_grids, cosmology):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grids : grids.CoordsCollection
            The image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        class MultiTracerGeometry(object):

            def __init__(self, planes_redshift_order, cosmology):

                self.final_plane = len(planes_redshift_order)-1

                self.plane_arcsec_per_kpc = list(map(lambda redshift : cosmology.arcsec_per_kpc_proper(z=redshift),
                                                     planes_redshift_order))

                self.plane_kpc_per_arcsec = list(map(lambda arcsec_per_kpc : 1.0 / arcsec_per_kpc,
                                                     self.plane_arcsec_per_kpc))

                self.angs_to_earth_kpc = list(map(lambda redshift :
                                                  cosmology.angular_diameter_distance(z=redshift).to('kpc'),
                                                  planes_redshift_order))

                self.angs_between_planes_kpc = []
                for plane_index in range(len(planes_redshift_order)):
                    self.angs_between_planes_kpc.append(list(map(lambda redshift :
                    cosmology.angular_diameter_distance_z1z2(planes_redshift_order[plane_index], redshift).to('kpc'),
                                                                         planes_redshift_order)))

                constant_kpc = (constants.c.to('kpc / s').value) ** 2.0 \
                               / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

                # TODO : Does this need generalizing to multi-planes? Do we need to distinguish thhe main 'lens plane'?

                self.critical_density_kpc = constant_kpc * self.angs_to_earth_kpc[1] / \
                                            (self.angs_between_planes_kpc[0][1] * self.angs_to_earth_kpc[0])

                self.critical_density_arcsec = self.critical_density_kpc * self.plane_kpc_per_arcsec[0] ** 2.0

            def compute_scaling_factor(self, plane_i, plane_j):

                return (self.angs_between_planes_kpc[plane_i][plane_j] * self.angs_to_earth_kpc[self.final_plane]) \
                       / (self.angs_to_earth_kpc[plane_j] * self.angs_between_planes_kpc[plane_i][self.final_plane])

        self.cosmology = cosmology
        self.galaxies_redshift_order = sorted(galaxies, key=lambda galaxy : galaxy.redshift, reverse=False)

        # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
        # Using a list of class attributes so make a list of redshifts for now.

        galaxy_redshifts = list(map(lambda galaxy : galaxy.redshift, self.galaxies_redshift_order))
        self.planes_redshift_order = [redshift for i, redshift in enumerate(galaxy_redshifts)
                                      if redshift not in galaxy_redshifts[:i]]

        if self.cosmology is not None:
            self.geometry = MultiTracerGeometry(self.planes_redshift_order, cosmology)
        else:
            self.geometry = None

        # TODO : Idea is to get a list of all galaxies in each plane - can you clean up the logic below?

        self.planes_galaxies = []

        for (plane_index, plane_redshift) in enumerate(self.planes_redshift_order):
            self.planes_galaxies.append(list(map(lambda galaxy :
                                        galaxy if galaxy.redshift == plane_redshift else None,
                                        self.galaxies_redshift_order)))
            self.planes_galaxies[plane_index] = list(filter(None, self.planes_galaxies[plane_index]))

        self.planes = []

        for plane_index in range(0, len(self.planes_redshift_order)):

            if plane_index < len(self.planes_redshift_order)-1:
                compute_deflections = True
            elif plane_index == len(self.planes_redshift_order)-1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_grid = image_plane_grids

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = self.geometry.compute_scaling_factor(plane_i=previous_plane_index,
                                                                          plane_j=plane_index)

                    scaled_deflections = self.planes[previous_plane_index].deflections. \
                        scaled_deflection_grids_for_scaling_factor(scaling_factor)

                    new_grid = new_grid.traced_grids_for_deflections(scaled_deflections)

            self.planes.append(Plane(galaxies=self.planes_galaxies[plane_index], grids=new_grid,
                                     compute_deflections=compute_deflections))


class Plane(object):

    def __init__(self, galaxies, grids, compute_deflections=True):
        """

        Represents a plane, which is a set of galaxies and grids at a given redshift in the lens ray-tracing
        calculation.

        The image-plane coordinates are defined on the observed image's uniform regular grid_coords. Calculating its
        model images from its light profiles exploits this uniformity to perform more efficient and precise calculations
        via an iterative sub-griding approach.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to
        the ImagePlane. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid_coords is preferred and
        won't lead noticeable inaccuracy. For example, computing the light profile of the main lens galaxy, ignoring
        minor lensing effects due to a low mass foreground substructure.

        2) When evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the
        point-source images of a lensed quasar, effects like micro-lensing mean lens-plane modeling will be inaccurate.


        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        grids : grids.GridCoordsCollection
            The grids of (x,y) coordinates in the plane, including the image grid_coords, sub-grid_coords, blurring,
            grid_coords, etc.
        """
        self.galaxies = galaxies
        self.grids = grids
        if compute_deflections:
            self.deflections = self.grids.deflection_grids_for_galaxies(self.galaxies)

    def trace_to_next_plane(self):
        """Trace the grids to the next plane.

        NOTE : This does not work for multi-plane lensing, which requires one to use the previous plane's deflection
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list.
        """
        return self.grids.traced_grids_for_deflections(self.deflections)

    def generate_image_of_galaxy_light_profiles(self, mapping):
        """Generate the image of the galaxies in this plane."""
        return self.grids.sub.intensities_via_grid(self.galaxies, mapping)

    def generate_blurring_image_of_galaxy_light_profiles(self):
        """Generate the image of the galaxies in this plane."""
        return self.grids.blurring.intensities_via_grid(self.galaxies)

    def generate_pixelization_matrices_of_galaxy(self, mapping):

        pixelized_galaxies = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(pixelized_galaxies) == 0:
            return None
        if len(pixelized_galaxies) == 1:
            return pixelized_galaxies[0].pixelization.inversion_from_pix_grids(self.grids.image,
                                                                               self.grids.sub, mapping)
        elif len(pixelized_galaxies) > 1:
            raise exc.PixelizationException('The number of galaxies with pixelizations in one plane is above 1')