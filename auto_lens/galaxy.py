import numpy as np
import matplotlib.pyplot as plt
import math
from astropy import cosmology
from astropy import constants

class LensingPlanes(list):

    def __init__(self, galaxies, cosmological_model=cosmology.Planck15):


        super().__init__()

        for galaxy in galaxies:
            self.append(galaxy)

        self.setup_angular_diameter_distances(cosmological_model)

        self.setup_critical_densities()

    def append(self, galaxy):
        """
        Append a new galaxy to the collection in the correct position according to its redshift.

        Parameters
        ----------
        galaxy: Galaxy
            A galaxy
        """

        def insert(position):
            if position == len(self):
                super(LensingPlanes, self).append(galaxy)
            elif galaxy.redshift <= self[position].redshift:
                self[:] = self[:position] + [galaxy] + self[position:]
            else:
                insert(position + 1)

        insert(0)

    def setup_angular_diameter_distances(self, cosmological_model):
        """Using the redshift of each galaxy, setup their angular diameter distances"""
        for i, galaxy in enumerate(self):

            galaxy.arcsec_per_kpc = cosmological_model.arcsec_per_kpc_proper(z=galaxy.redshift).value
            galaxy.kpc_per_arcsec = 1.0 / galaxy.arcsec_per_kpc

            galaxy.setup_angular_diameter_distance_to_earth(cosmological_model)

            if i < len(self) - 1:
                galaxy.setup_angular_diameter_distance_to_next_galaxy(cosmological_model, self[i + 1].redshift)

            if i > 0:

                galaxy.setup_angular_diameter_distance_to_previous_galaxy(cosmological_model, self[i - 1].redshift)

    def setup_critical_densities(self):
        """Setup the critical density of each galaxy."""

        # TODO : Don't know how to do this for multiple galaxies, so currently requires a standard 2 lens-source system.

        if len(self) == 2:

            for i, galaxy in enumerate(self):

                if i < len(self) - 1:
                    constant_kpc = (constants.c.to('kpc / s').value) ** 2.0 \
                                   / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

                    galaxy.critical_density_kpc = constant_kpc * self[i + 1].ang_to_earth_kpc \
                                                  / (galaxy.ang_to_next_galaxy_kpc *
                                                     galaxy.ang_to_earth_kpc)

                    galaxy.critical_density = galaxy.critical_density_kpc * galaxy.kpc_per_arcsec ** 2.0


class Galaxy(object):
    """Represents a real galaxy. This could be a lens galaxy or source galaxy. Note that a lens galaxy must have mass \
    profiles"""

    def __init__(self, redshift=None, light_profiles=None, mass_profiles=None, pixelization=None):
        """
        Parameters
        ----------
        redshift: float
            The redshift of this galaxy
        light_profile: LightProfile
            A list of light profiles describing the light profiles of this galaxy
        mass_profile: MassProfile
            A list of mass profiles describing the mass profiles of this galaxy
        """
        self.redshift = redshift
        self.light_profiles = light_profiles
        self.mass_profiles = mass_profiles
        self.pixelization = pixelization

        # TODO: All of the initial calls to an instance variable should be made in the constructor. self.ang_to_earth
        # TODO: etc. should be made here. However, it's still bad practice to be setting these variables to None at the
        # TODO: point of construction because much of the function of a class depends on them not being None.
        self.setup_cosmological_quantities()
        
    def setup_cosmological_quantities(self):
        # TODO: these shouldn't be in the Galaxy class. A Galaxy doesn't care about its angle to earth.

        self.ang_to_earth_kpc = None
        self.ang_to_next_galaxy_kpc = None
        self.ang_to_previous_galaxy_kpc = None
        self.ang_to_earth = None
        self.ang_to_next_galaxy = None
        self.ang_to_previous_galaxy = None
        self.arcsec_per_kpc = None
        self.kpc_per_arcsec = None

        # TODO: I still don't fully understand what critical density means. However, if it is something dependent on the
        # TODO: system as a whole (i.e. multiple galaxies) then it should not belong to any one galaxy. Either Critical
        # TODO: density is passed in as an argument or those functions that use it should not be in the galaxy class
        self.critical_density = None

    def setup_angular_diameter_distance_to_earth(self, cosmological_model):
        self.ang_to_earth_kpc = cosmological_model.angular_diameter_distance(z=self.redshift).to('kpc').value
        self.ang_to_earth = self.ang_to_earth_kpc * self.arcsec_per_kpc

    def setup_angular_diameter_distance_to_next_galaxy(self, cosmological_model, next_redshift):

        self.ang_to_next_galaxy_kpc = cosmological_model.angular_diameter_distance_z1z2(self.redshift,
                                                                                        next_redshift).to('kpc').value
        self.ang_to_next_galaxy = self.ang_to_next_galaxy_kpc * self.arcsec_per_kpc

    def setup_angular_diameter_distance_to_previous_galaxy(self, cosmological_model, previous_redshift):

        self.ang_to_previous_galaxy_kpc = cosmological_model.angular_diameter_distance_z1z2(previous_redshift,
                                                                                            self.redshift).to(
            'kpc').value
        self.ang_to_previous_galaxy = self.ang_to_previous_galaxy_kpc * self.arcsec_per_kpc

    def __repr__(self):
        return "<Galaxy redshift={}>".format(self.redshift)

    def intensity_at_coordinates(self, coordinates):
        """
        Compute the summed intensity of the galaxy's light profiles at a given set of coordinates.

        See *light_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The coordinates in image space
        Returns
        -------
        intensity : float
            The summed values of intensity at the given coordinates
        """
        return sum(map(lambda p : p.intensity_at_coordinates(coordinates), self.light_profiles))

    def intensity_at_coordinates_individual(self, coordinates):
        """
        Compute the individual intensities of the galaxy's light profiles at a given set of coordinates.

        See *light_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The coordinates in image space
        Returns
        -------
        intensity : float
            The summed values of intensity at the given coordinates
        """
        return list(map(lambda p : p.intensity_at_coordinates(coordinates), self.light_profiles))

    def luminosity_within_circle(self, radius):
        """
        Compute the total luminosity of the galaxy's light profiles within a circle of specified radius.

        See *light_profiles.luminosity_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the luminosity within.

        Returns
        -------
        luminosity : float
            The total combined luminosity within the specified circle.
        """
        return sum(map(lambda p : p.luminosity_within_circle(radius), self.light_profiles))

    def luminosity_within_circle_individual(self, radius):
        """
        Compute the individual total luminosity of each light profile in the galaxy, within a circle of
        specified radius.

        See *light_profiles.luminosity_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the luminosity within.

        Returns
        -------
        luminosity : float
            The total combined luminosity within the specified circle.
        """
        return list(map(lambda p : p.luminosity_within_circle(radius), self.light_profiles))

    def luminosity_within_ellipse(self, major_axis):
        """
        Compute the total luminosity of the galaxy's light profiles, within an ellipse of specified major axis. This \
        is performed via integration of each light profile and is centred, oriented and \ aligned with each light \
        model's individual geometry.

        See *light_profiles.luminosity_within_ellipse* for details of how this is performed.

        Parameters
        ----------
        major_axis: float
            The major-axis of the ellipse to compute the luminosity within.
        Returns
        -------
        intensity : float
            The total luminosity within the specified ellipse.
        """
        return sum(map(lambda p : p.luminosity_within_ellipse(major_axis), self.light_profiles))

    def luminosity_within_ellipse_individual(self, major_axis):
        """
        Compute the individual total luminosity of each light profile in the galaxy, within an ellipse of \
        specified major axis.

        See *light_profiles.luminosity_within_ellipse* for details of how this is performed.

        Parameters
        ----------
        major_axis: float
            The major-axis of the ellipse to compute the luminosity within.
        Returns
        -------
        intensity : float
            The total luminosity within the specified ellipse.
        """
        return list(map(lambda p : p.luminosity_within_ellipse(major_axis), self.light_profiles))

    def surface_density_at_coordinates(self, coordinates):
        """

        Compute the summed surface density of the galaxy's mass profiles at a given set of coordinates.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of surface density at the given coordinates.
        """
        return sum(map(lambda p : p.surface_density_at_coordinates(coordinates), self.mass_profiles))

    def surface_density_at_coordinates_individual(self, coordinates):
        """

        Compute the individual surface densities of the galaxy's mass profiles at a given set of coordinates.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of surface density at the given coordinates.
        """
        return list(map(lambda p : p.surface_density_at_coordinates(coordinates), self.mass_profiles))

    def potential_at_coordinates(self, coordinates):
        """
        Compute the summed gravitional potential of the galaxy's mass profiles at a given set of coordinates.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of gravitational potential at the given coordinates.
        """
        return sum(map(lambda p : p.potential_at_coordinates(coordinates), self.mass_profiles))

    def potential_at_coordinates_individual(self, coordinates):
        """
        Compute the individual gravitional potentials of the galaxy's mass profiles at a given set of coordinates.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of gravitational potential at the given coordinates.
        """
        return list(map(lambda p : p.potential_at_coordinates(coordinates), self.mass_profiles))

    def deflection_angles_at_coordinates(self, coordinates):
        """
        Compute the summed deflection angles of the galaxy's mass profiles at a given set of coordinates.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of deflection angles at the given coordinates.
        """
        sum_tuple = (0, 0)
        for t in map(lambda p: p.deflection_angles_at_coordinates(coordinates), self.mass_profiles):
            sum_tuple = (sum_tuple[0] + t[0], sum_tuple[1] + t[1])
        return sum_tuple

    def deflection_angles_at_coordinates_individual(self, coordinates):
        """
        Compute the individual deflection angles of the galaxy's mass profiles at a given set of coordinates.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of deflection angles at the given coordinates.
        """
        return list(map(lambda p : p.deflection_angles_at_coordinates(coordinates), self.mass_profiles))

    def mass_within_circles(self, radius):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.mass_within_circles* for details of how this is performed.


        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return self.critical_density * sum(map(lambda p: p.dimensionless_mass_within_circle(radius), self.mass_profiles))

    def mass_within_circles_individual(self, radius):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.mass_within_circles* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return self.critical_density * np.asarray(list(map(lambda p: p.dimensionless_mass_within_circle(radius), self.mass_profiles)))

    def mass_within_ellipses(self, major_axis):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        See *mass_profiles.mass_within_ellipses* for details of how this is performed.


        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return self.critical_density * sum(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles))

    def mass_within_ellipses_individual(self, major_axis):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within an ellipse of specified \
        major-axis.

        See *mass_profiles.mass_within_circles* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return self.critical_density * np.asarray(list(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles)))

    def plot_density_as_function_of_radius(self, maximum_radius, image_name='', labels=None, number_bins=50,
                                           xaxis_is_physical=True, yaxis_is_physical=True):
        """Produce a plot of the galaxy density as a function of radius.

        Parameters
        ----------
        maximum_radius : float
            The maximum radius the mass is plotted too.
        labels : [str]
            The label of each component for the legend
        number_bins : int
            The number of bins used to compute and plot the mass.
        """

        radii = list(np.linspace(1e-4, maximum_radius, number_bins+1))

        for i in range(number_bins):

            annuli_area = (math.pi*radii[i+1]**2 - math.pi*radii[i]**2)

            densities = ((self.mass_within_circles_individual(radii[i + 1]) -
                           self.mass_within_circles_individual(radii[i])) /
                           annuli_area)

        plt.title('Decomposed surface density profile of ' + image_name, size=16)

        if xaxis_is_physical:
            radii_plot = list(np.linspace(1e-4, maximum_radius*self.kpc_per_arcsec, number_bins))
            plt.xlabel('Distance From Galaxy Center (kpc)', size=16)
        else:
            radii_plot = list(np.linspace(1e-4, maximum_radius, number_bins))
            plt.xlabel('Distance From Galaxy Center (")', size=16)

        if yaxis_is_physical:
            plt.ylabel(r'Surface Mass Density $\Sigma$ ($\frac{M_{odot}}{kpc^2}})$')
        else:
            pass

        plt.semilogy(radii_plot, densities, color='r', label='Sersic Bulge')

        # plt.axvline(x=self.einstein_radius*self.kpc_per_arcsec, linestyle='--')
        # plt.axvline(x=self.source_light_min*self.kpc_per_arcsec, linestyle='-')
        # plt.axvline(x=self.source_light_max*self.kpc_per_arcsec, linestyle='-')

        plt.legend(labels)
        plt.show()