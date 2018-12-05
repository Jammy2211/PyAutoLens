import math

import numpy as np
from astropy import constants
from astropy import cosmology as cosmo

from autolens import exc
from autolens.data.array import mask as msk
from autolens.lensing import plane as pl


class AbstractTracer(object):
    image_plane_grids = None

    def __init__(self, image_plane):
        self.image_plane = image_plane

    @property
    def all_planes(self):
        raise NotImplementedError()

    @property
    def total_planes(self):
        return len(self.all_planes)

    @property
    def total_images(self):
        return len(self.image_plane.grids)

    @property
    def redshifts(self):
        return [plane.redshift for plane in self.all_planes]

    @property
    def has_light_profile(self):
        return any(list(map(lambda plane: plane.has_light_profile, self.all_planes)))

    @property
    def has_pixelization(self):
        return any(list(map(lambda plane: plane.has_pixelization, self.all_planes)))

    @property
    def has_regularization(self):
        return any(list(map(lambda plane: plane.has_regularization, self.all_planes)))

    @property
    def has_padded_grids(self):
        return isinstance(self.all_planes[0].grids.regular, msk.PaddedImageGrid)

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda plane: plane.has_hyper_galaxy, self.all_planes)))

    @property
    def hyper_galaxies(self):
        return list(filter(None, [hyper_galaxy for plane in self.all_planes for hyper_galaxy in plane.hyper_galaxies]))

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    @property
    def image_plane_images(self):
        return list(map(lambda _image_plane_image, grid : grid.regular.scaled_array_from_array_1d(_image_plane_image),
                        self.image_plane_images_, self.image_plane.grids))

    @property
    def image_plane_images_for_simulation(self):
        return[sum(image_plane_image_of_plane_for_simulation[i] for image_plane_image_of_plane_for_simulation in
                   self.image_plane_images_of_planes_for_simulation) for i in range(self.total_images)]

    @property
    def image_plane_images_of_planes_for_simulation(self):
        return [plane.image_plane_images_for_simulation for plane in self.all_planes]

    @property
    def mappers_of_planes(self):
        return list(filter(None, [plane.mapper for plane in self.all_planes]))

    @property
    def regularizations_of_planes(self):
        return list(filter(None, [plane.regularization for plane in self.all_planes]))

    @property
    def image_plane_images_(self):
        return list(map(lambda _image_plane_images_of_planes : sum(_image_plane_images_of_planes),
                        self.image_plane_images_of_planes_))

    # TODO : It makes the high level code a lot more intuitive if the indexing of regular of planes goes
    # TODO : [image_index][plane_index]. Is there a neat dictionary comprehension to do this rather than the loop below?

    @property
    def image_plane_images_of_planes_(self):
        image_plane_images_ = [plane.image_plane_images_ for plane in self.all_planes]
        image_plane_images_of_planes_ = [[] for _ in range(self.total_images)]
        for image_index in range(self.total_images):
            for plane_index in range(self.total_planes):
                image_plane_images_of_planes_[image_index].append(image_plane_images_[plane_index][image_index])
        return image_plane_images_of_planes_

    @property
    def image_plane_blurring_images_(self):
        return list(map(lambda _image_plane_blurring_images_of_planes : sum(_image_plane_blurring_images_of_planes),
                        self.image_plane_blurring_images_of_planes_))

    @property
    def image_plane_blurring_images_of_planes_(self):
        image_plane_blurring_images_ = [plane.image_plane_blurring_images_ for plane in self.all_planes]
        image_plane_blurring_images_of_planes_ = [[] for _ in range(self.total_images)]
        for image_index in range(self.total_images):
            for plane_index in range(self.total_planes):
                image_plane_blurring_images_of_planes_[image_index].append(image_plane_blurring_images_[plane_index][image_index])
        return image_plane_blurring_images_of_planes_

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

    @property
    def image_plane_image(self):
        return self.image_plane_images[0]

    @property
    def image_plane_image_for_simulation(self):
        return self.image_plane_images_for_simulation[0]


class TracerImagePlane(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane]

    def __init__(self, lens_galaxies, image_plane_grids, border=None, cosmology=None):
        """Ray-tracer_normal for a lensing system with just one plane, the datas_-plane. Because there is 1 plane, there are \
        no ray-tracing calculations and the class is used purely for fitting datas_-plane galaxies with light \
        profiles.
        
        By default, this has no associated cosmology and galaxy quantities (e.g. effective radii) are in \
        arc-seconds. If a cosmology is supplied, the plane's angular diameter distances, conversion factors, etc. \
        are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the datas_-plane.
        image_plane_grids : grids.DataGrids
            The datas_-plane grids where tracer_normal calculation are performed, (this includes the datas_-grid, sub-grid, \
            blurring-grid, etc.).
        border : mask.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
            source-plane border.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        if not lens_galaxies:
            raise exc.RayTracingException('No lens galaxies have been input into the Tracer')

        super().__init__(pl.Plane(lens_galaxies, image_plane_grids, border=border, compute_deflections=True,
                                  cosmology=cosmology))


class TracerImageSourcePlanes(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grids, border=None, cosmology=None):
        """Ray-tracer_normal for a lensing system with two planes, an datas_-plane and source-plane.

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. If a cosmology is supplied, the plane's angular diameter distances, \ 
        conversion factors, etc. are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the datas_-plane.
        source_galaxies : [Galaxy]
            The list of galaxies in the source-plane.
        image_plane_grids : grids.DataGrids
            The datas_-plane grids where ray-tracing calculation are performed, (this includes the datas_-grid, \
            sub-grid, blurring-grid, etc.).
        border : mask.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
            source-plane border.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        super().__init__(pl.Plane(lens_galaxies, image_plane_grids, border=border, compute_deflections=True,
                                  cosmology=cosmology))

        self.cosmology = cosmology

        source_plane_grids = self.image_plane.trace_grids_to_next_plane()

        self.source_plane = pl.Plane(source_galaxies, source_plane_grids, border=border, compute_deflections=False,
                                     cosmology=cosmology)

    @property
    @pl.cosmology_check
    def angular_diameter_distance_from_image_to_source_plane(self):
        return self.cosmology.angular_diameter_distance_z1z2(self.image_plane.redshift,
                                                             self.source_plane.redshift).to('kpc').value

    @property
    @pl.cosmology_check
    def critical_density_kpc(self):
        return self.constant_kpc * self.source_plane.angular_diameter_distance_to_earth / \
               (self.angular_diameter_distance_from_image_to_source_plane *
                self.image_plane.angular_diameter_distance_to_earth)

    @property
    @pl.cosmology_check
    def critical_density_arcsec(self):
        return self.critical_density_kpc * self.image_plane.kpc_per_arcsec_proper ** 2.0

    @property
    def source_plane_image(self):
        return self.plane_images_of_planes(shape=(50, 50))

    def masses_of_image_plane_galaxies_within_circles(self, radius):
        """
        Compute the total mass of all galaxies in the regular-plane within a circle of specified radius, using the plane's
        critical surface density to convert this to physical units.

        For a single galaxy, inputting the Einstein Radius should provide an accurate measurement of the Einstein mass.
        Use of other radii may be subject to systematic offsets, because lensing does not directly measure the mass of
        a galaxy beyond the Einstein radius.

        For multiple galaxies, the Einstein mass of the entire regular-plane is evenly divided across its galaxies. This
        could be highly inaccurate and users are recommended to cross-check mass estimates using different radii.

        See *galaxy.dimensionless_mass_within_circle* and *mass_profiles.dimensionless_mass_within_circle* for details
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

        For a single galaxy, inputting the Einstein Radius should provide an accurate measurement of the Einstein mass.
        Use of other radii may be subject to systematic offsets, because lensing does not directly measure the mass of
        a galaxy beyond the Einstein radius.

        For multiple galaxies, the Einstein mass of the entire regular-plane is evenly divided across its galaxies. This
        could be highly inaccurate and users are recommended to cross-check mass estimates using different radii.

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


class AbstractTracerMulti(AbstractTracer):

    def __init__(self, galaxies, cosmology):
        """The ray-tracing calculations, defined by a lensing system with just one datas_-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
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
        return [p for p in self.planes]

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
               (self.angular_diameter_distance_between_planes(i, j) * self.angular_diameter_distance_of_plane_to_earth(
                   i))

    def critical_density_arcsec_between_planes(self, i, j):
        return self.critical_density_kpc_between_planes(i, j) * self.kpc_per_arcsec_proper_of_plane(i) ** 2.0

    def scaling_factor_between_planes(self, i, j):
        return (self.angular_diameter_distance_between_planes(i, j) *
                self.angular_diameter_distance_to_source_plane) / \
               (self.angular_diameter_distance_of_plane_to_earth(j) *
                self.angular_diameter_distance_between_planes(i, self.source_plane_index))


class TracerMulti(AbstractTracerMulti):

    def __init__(self, galaxies, image_plane_grids, border=None, cosmology=cosmo.Planck15):
        """Ray-tracer_normal for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lensing-geometry of the multi-plane system.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grids : grids.DataGrids
            The datas_-plane grids where ray-tracing calculation are performed, (this includes the
            datas_-grid, sub-grid, blurring-grid, etc.).
        border : mask.RegularGridBorder
            The border of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
            source-plane border.
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

            new_grids = image_plane_grids

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = self.scaling_factor_between_planes(i=previous_plane_index, j=plane_index)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    if self.planes[previous_plane_index].deflections is not None:
                        scaled_deflections = list(map(lambda deflections : deflections.apply_function(scale),
                                                      self.planes[previous_plane_index].deflections))
                    else:
                        scaled_deflections = None

                    if scaled_deflections is not None:

                        def minus(grid, deflections):
                            return grid - deflections

                        new_grids = list(map(lambda grid, deflections: grid.map_function(minus, deflections),
                                 new_grids, scaled_deflections))

            self.planes.append(pl.Plane(galaxies=self.planes_galaxies[plane_index], grids=new_grids, border=border,
                                        compute_deflections=compute_deflections, cosmology=cosmology))


class TracerImageSourcePlanesPositions(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, positions, cosmology=None):
        """Positional ray-tracer_normal for a lensing system with two planes, an datas_-plane and source-plane (source-plane \
        galaxies are not input for the positional ray-tracer_normal, as it is only the proximity that positions trace to \
        within one another that needs to be computed).

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. If a cosmology is supplied, the plane's angular diameter distances, \
        conversion factors, etc. are used to provide quantities in kpc.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the datas_-plane.
        positions : [[[]]]
            The (x,y) arc-second coordinates of datas_-plane pixels which (are expected to) mappers to the same location(s) \
            in the source-plane.
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        super().__init__(pl.PlanePositions(lens_galaxies, positions, compute_deflections=True, cosmology=cosmology))

        self.cosmology = cosmology

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = pl.PlanePositions(None, source_plane_grids, compute_deflections=False,
                                              cosmology=cosmology)


class TracerMultiPositions(AbstractTracerMulti):

    def __init__(self, galaxies, positions, cosmology):
        """Positional ray-tracer_normal for a lensing system with any number of planes.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lensing-geometry of the multi-plane system.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        positions : [[[]]]
            The (x,y) arc-second coordinates of datas_-plane pixels which (are expected to) mappers to the same location(s) \
            in the final source-plane.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcepl.Planes)')

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

            self.planes.append(pl.PlanePositions(galaxies=self.planes_galaxies[plane_index], positions=new_positions,
                                                 compute_deflections=compute_deflections))
