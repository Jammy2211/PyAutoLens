import numpy as np
import matplotlib.pyplot as plt
import math

# class Cosmology(object):

#    def __init__(self):


class GalaxyCollection(list):
    """A collection of galaxies ordered by redshift"""

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
                super(GalaxyCollection, self).append(galaxy)
            elif galaxy.redshift <= self[position].redshift:
                self[:] = self[:position] + [galaxy] + self[position:]
            else:
                insert(position + 1)

        insert(0)


class Galaxy(object):
    """Represents a real galaxy. This could be a lens galaxy or source galaxy. Note that a lens galaxy must have mass \
    profiles"""

    def __init__(self, redshift, light_profiles=None, mass_profiles=None):
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

    def __repr__(self):
        return "<Galaxy redshift={}>".format(self.redshift)

    def intensity_at_coordinates(self, coordinates):
        """
        Compute the summed intensity of the galaxy's light profiles at a given set of coordinates.

        See *light_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : (float, float)
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
        coordinates : (float, float)
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
        coordinates : (float, float)
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
        coordinates : (float, float)
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
        coordinates : (float, float)
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
        coordinates : (float, float)
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
        coordinates : (float, float)
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

    def dimensionless_mass_within_circle(self, radius):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.dimensionless_mass_within_circle* for details of how this is performed.


        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return sum(map(lambda p: p.dimensionless_mass_within_circle(radius), self.mass_profiles))

    def dimensionless_mass_within_circle_individual(self, radius):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.dimensionless_mass_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return np.asarray(list(map(lambda p: p.dimensionless_mass_within_circle(radius), self.mass_profiles)))

    def dimensionless_mass_within_ellipse(self, major_axis):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        See *mass_profiles.dimensionless_mass_within_ellipse* for details of how this is performed.


        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return sum(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles))

    def dimensionless_mass_within_ellipse_individual(self, major_axis):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within an ellipse of specified \
        major-axis.

        See *mass_profiles.dimensionless_mass_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return np.asarray(list(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles)))

    def plot_density_as_function_of_radius(self, maximum_radius, labels=None, number_bins=50):
        """Produce a plot of the galaxy density as a function of radius.

        Parameters
        ----------
        maximum_radius : float
            The maximum radius the mass is plotted too.
        number_bins : int
            The number of bins used to compute and plot the mass.
        """

        radii = list(np.linspace(1e-4, maximum_radius, number_bins+1))

        density = []
        for i in range(number_bins):

            annuli_area = (math.pi*radii[i+1]**2 - math.pi*radii[i]**2)

            density.append((self.dimensionless_mass_within_circle_individual(radii[i+1]) -
                           self.dimensionless_mass_within_circle_individual(radii[i])) /
                           annuli_area)

        plt.semilogy(radii[0:-1], density)
        plt.legend(labels)
        plt.show()