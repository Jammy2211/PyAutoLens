import numpy as np
import matplotlib.pyplot as plt
import math


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

    def __repr__(self):
        return "<Galaxy redshift={}>".format(self.redshift)

    def intensity_at_coordinates(self, coordinates):
        """
        Compute the summed intensity of the galaxy's light profiles at a given set of image_grid.

        See *light_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The image_grid in image_grid space
        Returns
        -------
        intensity : float
            The summed values of intensity at the given image_grid
        """
        if self.light_profiles is not None:
            return sum(map(lambda p: p.intensity_at_coordinates(coordinates), self.light_profiles))
        else:
            return 0.0

    def intensity_at_coordinates_individual(self, coordinates):
        """
        Compute the individual intensities of the galaxy's light profiles at a given set of image_grid.

        See *light_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : (float, float)
            The image_grid in image_grid space
        Returns
        -------
        intensity : [float]
            The summed values of intensity at the given image_grid
        """
        return list(map(lambda p: p.intensity_at_coordinates(coordinates), self.light_profiles))

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
        return sum(map(lambda p: p.luminosity_within_circle(radius), self.light_profiles))

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
        luminosity : [float]
            The total combined luminosity within the specified circle.
        """
        return list(map(lambda p: p.luminosity_within_circle(radius), self.light_profiles))

    def luminosity_within_ellipse(self, major_axis):
        """
        Compute the total luminosity of the galaxy's light profiles, within an ellipse of specified major axis. This \
        is performed via integration of each light profile and is centred, oriented and \ aligned with each light \
        model_mapper's individual geometry.

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
        return sum(map(lambda p: p.luminosity_within_ellipse(major_axis), self.light_profiles))

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
        return list(map(lambda p: p.luminosity_within_ellipse(major_axis), self.light_profiles))

    def surface_density_at_coordinates(self, coordinates):
        """

        Compute the summed surface density of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of surface density at the given image_grid.
        """
        return sum(map(lambda p: p.surface_density_at_coordinates(coordinates), self.mass_profiles))

    def surface_density_at_coordinates_individual(self, coordinates):
        """

        Compute the individual surface densities of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of surface density at the given image_grid.
        """
        return list(map(lambda p: p.surface_density_at_coordinates(coordinates), self.mass_profiles))

    def potential_at_coordinates(self, coordinates):
        """
        Compute the summed gravitional potential of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of gravitational potential at the given image_grid.
        """
        return sum(map(lambda p: p.potential_at_coordinates(coordinates), self.mass_profiles))

    def potential_at_coordinates_individual(self, coordinates):
        """
        Compute the individual gravitional potentials of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of gravitational potential at the given image_grid.
        """
        return list(map(lambda p: p.potential_at_coordinates(coordinates), self.mass_profiles))

    def deflections_at_coordinates(self, coordinates):
        """
        Compute the summed deflection angles of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of deflection angles at the given image_grid.
        """
        if self.mass_profiles is not None:
            return sum(map(lambda p: p.deflections_at_coordinates(coordinates), self.mass_profiles))
        else:
            return np.array([0.0, 0.0])

    def deflection_angles_at_coordinates_individual(self, coordinates):
        """
        Compute the individual deflection angles of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of deflection angles at the given image_grid.
        """
        return np.asarray(list(map(lambda p: p.deflections_at_coordinates(coordinates), self.mass_profiles)))

    def dimensionless_mass_within_circles(self, radius):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.dimensionless_mass_within_circles* for details of how this is performed.


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

    def dimensionless_mass_within_circles_individual(self, radius):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.dimensionless_mass_within_circles* for details of how this is performed.

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

    def dimensionless_mass_within_ellipses(self, major_axis):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        See *mass_profiles.dimensionless_mass_within_ellipses* for details of how this is performed.


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

    def dimensionless_mass_within_ellipses_individual(self, major_axis):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within an ellipse of specified \
        major-axis.

        See *mass_profiles.dimensionless_mass_within_circles* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return list(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles))

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

        # TODO: This function seems to be broken

        radii = list(np.linspace(1e-4, maximum_radius, number_bins + 1))

        for i in range(number_bins):
            annuli_area = (math.pi * radii[i + 1] ** 2 - math.pi * radii[i] ** 2)

            # TODO: Densities is being replaced each iteration of the loop??
            densities = ((self.mass_within_circles_individual(radii[i + 1]) -
                          self.mass_within_circles_individual(radii[i])) /
                         annuli_area)

        plt.title('Decomposed surface density profile of ' + image_name, size=16)

        if xaxis_is_physical:
            radii_plot = list(np.linspace(1e-4, maximum_radius * self.kpc_per_arcsec, number_bins))
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
