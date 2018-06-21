import numpy as np


class Galaxy(object):
    """
    Represents a real galaxy. This could be a lens galaxy or source galaxy. Note that a lens galaxy must have mass
    profiles
    """

    def __init__(self, redshift=None, light_profiles=None, mass_profiles=None, pixelization=None):
        """
        Parameters
        ----------
        redshift: float
            The redshift of this galaxy
        light_profiles: [LightProfile]
            A list of light profiles describing the light profiles of this galaxy
        mass_profiles: [MassProfile]
            A list of mass profiles describing the mass profiles of this galaxy
        """
        self.redshift = redshift
        self.light_profiles = light_profiles if light_profiles is not None else []
        self.mass_profiles = mass_profiles if mass_profiles is not None else []
        self.pixelization = pixelization

    @property
    def has_pixelization(self):
        return self.pixelization is not None

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
        return sum(map(lambda p: p.intensity_at_coordinates(coordinates), self.light_profiles))

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
        Compute the total luminosity of the galaxy's light profiles, within an ellipse of specified major axis. This 
        is performed via integration of each light profile and is centred, oriented and  aligned with each light 
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
        Compute the individual total luminosity of each light profile in the galaxy, within an ellipse of 
        specified major axis.

        See *light_profiles.luminosity_within_ellipse* for details of how this is performed.

        Parameters
        ----------
        major_axis: float
            The major-axis of the ellipse to compute the luminosity within.
        Returns
        -------
        intensity : [float]
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
        Compute the summed gravitational potential of the galaxy's mass profiles at a given set of image_grid.

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
        Compute the individual gravitational potentials of the galaxy's mass profiles at a given set of image_grid.

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
        major_axis : float
            The major axis of the ellipse

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return sum(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles))

    def dimensionless_mass_within_ellipses_individual(self, major_axis):
        """
        Compute the individual dimensionless mass of the galaxy's mass profiles within an ellipse of specified 
        major-axis.

        See *mass_profiles.dimensionless_mass_within_circles* for details of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major axis of the ellipse

        Returns
        -------
        dimensionless_mass : float
            The total dimensionless mass within the specified circle.
        """
        return list(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles))


# TODO : Is should galaxy image and minimum value be in the constructor (they arn't free parameters)?

class HyperGalaxy(object):

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        """Class for scaling the noise in the different galaixe of an image (e.g. the lens, source).

        Parameters
        -----------
        contribution_factor : float
            Factor that adjusts how much of the galaxy's light is attributed to the contribution map.
        noise_factor : float
            Factor by which the noise is increased in the regions of the galaxy's contribution map.
        noise_power : float
            The power to which the contribution map is raised when scaling the noise.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

    def compute_contributions(self, model_image, galaxy_image, minimum_value):
        """Compute the contribution map of a galaxy, which represents the fraction of flux in each pixel that \
        galaxy can be attributed to contain.

        This is computed by dividing that galaxy's flux by the total flux in that pixel, and then scaling by the \
        maximum flux such that the contribution map ranges between 0 and 1.

        Parameters
        -----------
        model_image : ndarray
            The model image of the observed data (from a previous analysis phase). This tells us the total light \
            attributed to each image pixel by the model.
        galaxy_image : ndarray
            A model image of the galaxy (e.g the lens light profile or source reconstruction) computed from a
            previous analysis.
        minimum_value : float
            The minimum fractional flux a pixel must contain to not be rounded to 0.
        """
        contributions = galaxy_image / (model_image + self.contribution_factor)
        contributions = contributions / np.max(contributions)
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def compute_scaled_noise(self, noise, contributions):
        """Compute a scaled galaxy noise map from a baseline nosie map.

        This uses the galaxy contribution map with their noise scaling hyper-parameters.

        Parameters
        -----------
        noise : ndarray
            The noise before scaling (this may already have the background scaled in HyperImage)
        contributions : ndarray
            The galaxy contribution map.
        """
        return noise + (self.noise_factor * (noise * contributions) ** self.noise_power)


class Redshift(object):
    def __init__(self, redshift):
        self.redshift = redshift
