from itertools import count

import numpy as np

from autolens import exc
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp


def is_light_profile(obj):
    return isinstance(obj, lp.LightProfile)


def is_mass_profile(obj):
    return isinstance(obj, mp.MassProfile)


class Galaxy(object):
    """
    @DynamicAttrs
    """

    def __init__(self, redshift=None, pixelization=None, regularization=None, hyper_galaxy=None, **kwargs):
        """
        Represents a real galaxy. This could be a lens galaxy or source galaxy. Note that a lens galaxy must have mass
        profiles

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

        for name, val in kwargs.items():
            setattr(self, name, val)

        self.pixelization = pixelization
        self.regularization = regularization

        if self.has_pixelization and not self.has_regularization:
            raise exc.GalaxyException('If the galaxy has a pixelization, it must also have a regularization.')
        if not self.has_pixelization and self.has_regularization:
            raise exc.GalaxyException('If the galaxy has a regularization, it must also have a pixelization.')

        self.hyper_galaxy = hyper_galaxy

    @property
    def light_profiles(self):
        return [value for value in self.__dict__.values() if is_light_profile(value)]

    @property
    def mass_profiles(self):
        return [value for value in self.__dict__.values() if is_mass_profile(value)]

    @property
    def has_redshift(self):
        return self.redshift is not None

    @property
    def has_pixelization(self):
        """
        Returns
        -------
        True iff this galaxy has an associated inversion
        """
        return self.pixelization is not None

    @property
    def has_regularization(self):
        """
        Returns
        -------
        True iff this galaxy has an associated inversion
        """
        return self.regularization is not None

    @property
    def has_hyper_galaxy(self):
        """
        Returns
        -------
        True iff this galaxy has an associated hyper galaxy
        """
        return self.hyper_galaxy is not None

    @property
    def has_light_profile(self):
        """
        Returns
        -------
        True iff there is one or more mass or light profiles associated with this galaxy
        """
        return len(self.light_profiles) > 0

    @property
    def has_mass_profile(self):
        """
        Returns
        -------
        True iff there is one or more mass or light profiles associated with this galaxy
        """
        return len(self.mass_profiles) > 0

    @property
    def has_profile(self):
        """
        Returns
        -------
        True iff there is one or more mass or light profiles associated with this galaxy
        """
        return len(self.mass_profiles) + len(self.light_profiles) > 0

    def __repr__(self):
        string = "Redshift: {}".format(self.redshift)
        if self.pixelization:
            string += "\nPixelization:\n{}".format(str(self.pixelization))
        if self.hyper_galaxy:
            string += "\nHyper Galaxy:\n{}".format(str(self.hyper_galaxy))
        if self.light_profiles:
            string += "\nLight Profiles:\n{}".format("\n".join(map(str, self.light_profiles)))
        if self.mass_profiles:
            string += "\nMass Profiles:\n{}".format("\n".join(map(str, self.mass_profiles)))
        return string

    def intensities_from_grid(self, grid):
        if self.light_profiles is not None and len(self.light_profiles) > 0:
            return sum(map(lambda p: p.intensities_from_grid(grid), self.light_profiles))
        else:
            return np.zeros((grid.shape[0],))

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

    def luminosity_within_ellipse(self, major_axis):
        """
        Compute the total luminosity of the galaxy's light profiles, within an ellipse of specified major axis. This 
        is performed via integration_old of each light profile and is centred, oriented and  aligned with each light
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

    def surface_density_from_grid(self, grid):
        """

        Compute the summed surface density of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of surface density at the given image_grid.
        """
        if self.mass_profiles is not None and len(self.mass_profiles) > 0:
            return sum(map(lambda p: p.surface_density_from_grid(grid), self.mass_profiles))
        else:
            return np.zeros((grid.shape[0],))

    def potential_from_grid(self, grid):
        """
        Compute the summed gravitational potential of the galaxy's mass profiles at a given set of image_grid.

        See *mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : ndarray
            The x and y image_grid of the image_grid

        Returns
        ----------
        The summed values of gravitational potential at the given image_grid.
        """
        if self.mass_profiles is not None and len(self.mass_profiles) > 0:
            return sum(map(lambda p: p.potential_from_grid(grid), self.mass_profiles))
        else:
            return np.zeros((grid.shape[0],))

    def deflections_from_grid(self, grid):
        if self.mass_profiles is not None and len(self.mass_profiles) > 0:
            return sum(map(lambda p: p.deflections_from_grid(grid), self.mass_profiles))
        else:
            return np.full((grid.shape[0], 2), 0.0)

    def dimensionless_mass_within_circle(self, radius):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.dimensionless_mass_within_circle* for details of how this is performed.


        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        """
        return sum(map(lambda p: p.dimensionless_mass_within_circle(radius), self.mass_profiles))

    def dimensionless_mass_within_ellipse(self, major_axis):
        """
        Compute the total dimensionless mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        See *mass_profiles.dimensionless_mass_within_ellipses* for details of how this is performed.


        Parameters
        ----------
        major_axis : float
            The major axis of the ellipse.
        """
        return sum(map(lambda p: p.dimensionless_mass_within_ellipse(major_axis), self.mass_profiles))

    def mass_within_circle(self, radius, conversion_factor):
        """
        Compute the total mass of the galaxy's mass profiles within a circle of specified radius.

        See *mass_profiles.mass_within_circle* for details of how this is performed.


        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        conversion_factor : float
            The factor the dimensionless mass is multiplied by to convert it to a physical mass.
        """
        return sum(map(lambda p: p.mass_within_circle(radius, conversion_factor), self.mass_profiles))

    def mass_within_ellipse(self, major_axis, conversion_factor):
        """
        Compute the total mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        See *mass_profiles.mass_within_ellipses* for details of how this is performed.


        Parameters
        ----------
        major_axis : float
            The major axis of the ellipse
        conversion_factor : float
            The factor the dimensionless mass is multiplied by to convert it to a physical mass.
        """
        return sum(map(lambda p: p.mass_within_ellipse(major_axis, conversion_factor), self.mass_profiles))


class HyperGalaxy(object):
    _ids = count()

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        """Class for scaling the noise_map_ in the different galaxies of an masked_image (e.g. the lens, source).

        Parameters
        -----------
        contribution_factor : float
            Factor that adjusts how much of the galaxy's light is attributed to the contribution mappers.
        noise_factor : float
            Factor by which the noise_map_ is increased in the regions of the galaxy's contribution mappers.
        noise_power : float
            The power to which the contribution mappers is raised when scaling the noise_map_.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

        self.component_number = next(self._ids)

    def contributions_from_hyper_images(self, hyper_model_image, hyper_galaxy_image, minimum_value):
        """Compute the contribution mappers of a galaxy, which represents the fraction of flux in each pixel that \
        galaxy can be attributed to contain.

        This is computed by dividing that galaxy's flux by the total flux in that pixel, and then scaling by the \
        maximum flux such that the contribution mappers ranges between 0 and 1.

        Parameters
        -----------
        hyper_model_image : ndarray
            The model masked_image of the observed data_vector (from a previous lensing phase). This tells us the total
            light attributed to each masked_image pixel by the model.
        hyper_galaxy_image : ndarray
            A model masked_image of the galaxy (e.g the lens light profile or source reconstructed_image) computed from
            a previous lensing.
        minimum_value : float
            The minimum fractional flux a pixel must contain to not be rounded to 0.
        """
        contributions = np.divide(hyper_galaxy_image, np.add(hyper_model_image, self.contribution_factor))
        contributions = np.divide(contributions, np.max(contributions))
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def scaled_noise_from_contributions(self, noise, contributions):
        """Compute a scaled galaxy noise_map_ mappers from a baseline nosie mappers.

        This uses the galaxy contribution mappers with their noise_map_ scaling hyper-parameters.

        Parameters
        -----------
        noise : ndarray
            The noise_map_ before scaling (this may already have the background scaled in HyperImage)
        contributions : ndarray
            The galaxy contribution mappers.
        """
        return self.noise_factor * (noise * contributions) ** self.noise_power

    def __eq__(self, other):
        if isinstance(other, HyperGalaxy):
            return self.contribution_factor == other.contribution_factor and \
                   self.noise_factor == other.noise_factor and \
                   self.noise_power == other.noise_power
        return False

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])


class Redshift(object):
    def __init__(self, redshift):
        self.redshift = redshift

    def __str__(self):
        return str(self.redshift)
