from itertools import count

import numpy as np

from autolens import exc
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


def is_light_profile(obj):
    return isinstance(obj, lp.LightProfile)


def is_mass_profile(obj):
    return isinstance(obj, mp.MassProfile)


class Galaxy(object):
    """
    @DynamicAttrs
    """

    def __init__(self, redshift=None, pixelization=None, regularization=None, hyper_galaxy=None, **kwargs):
        """Class representing a galaxy, which is composed of attributes used for fitting hyper (e.g. light profiles, \ 
        mass profiles, pixelizations, etc.).
        
        All *has_* methods retun *True* if galaxy has that attribute, *False* if not.

        Parameters
        ----------
        redshift: float
            The redshift of the galaxy.
        light_profiles: [lp.LightProfile]
            A list of the galaxy's light profiles.
        mass_profiles: [mp.MassProfile]
            A list of the galaxy's mass profiles.
        hyper_galaxy : HyperGalaxy
            The hyper-parameters of the hyper-galaxy, which is used for performing a hyper-analysis on the noise-map.
            
        Attributes
        ----------
        pixelization : inversion.Pixelization
            The pixelization of the galaxy used to reconstruct an observed image using an inversion.
        regularization : inversion.Regularization
            The regularization of the pixel-grid used to reconstruct an observed regular using an inversion.
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
        return self.pixelization is not None

    @property
    def has_regularization(self):
        return self.regularization is not None

    @property
    def has_hyper_galaxy(self):
        return self.hyper_galaxy is not None

    @property
    def has_light_profile(self):
        return len(self.light_profiles) > 0

    @property
    def has_mass_profile(self):
        return len(self.mass_profiles) > 0

    @property
    def has_profile(self):
        return len(self.mass_profiles) + len(self.light_profiles) > 0

    def __repr__(self):
        string = "Redshift: {}".format(self.redshift)
        if self.pixelization:
            string += "\nPixelization:\n{}".format(str(self.pixelization))
        if self.regularization:
            string += "\nRegularization:\n{}".format(str(self.regularization))
        if self.hyper_galaxy:
            string += "\nHyper Galaxy:\n{}".format(str(self.hyper_galaxy))
        if self.light_profiles:
            string += "\nLight Profiles:\n{}".format("\n".join(map(str, self.light_profiles)))
        if self.mass_profiles:
            string += "\nMass Profiles:\n{}".format("\n".join(map(str, self.mass_profiles)))
        return string

    def __eq__(self, other):
        return all((isinstance(other, Galaxy),
                    self.pixelization == other.pixelization,
                    self.redshift == other.redshift,
                    self.hyper_galaxy == other.hyper_galaxy,
                    self.light_profiles == other.light_profiles,
                    self.mass_profiles == other.mass_profiles))

    def intensities_from_grid(self, grid):
        """Calculate the summed intensities of all of the galaxy's light profiles using a grid of Cartesian (y,x) \
        coordinates.
        
        If the galaxy has no light profiles, a grid of zeros is returned.
        
        See *profiles.light_profiles* for a description of how light profile intensities are computed.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_light_profile:
            return sum(map(lambda p: p.intensities_from_grid(grid), self.light_profiles))
        else:
            return np.zeros((grid.shape[0],))

    def luminosity_within_circle(self, radius, conversion_factor=1.0):
        """
        Compute the total luminosity of the galaxy's light profiles within a circle of specified radius.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the photometric zeropoint).

        See *light_profiles.luminosity_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the luminosity within.
        conversion_factor : float
            Factor the dimensionless luminosity is multiplied by to convert it to a physical luminosity \
            (e.g. a photometric zeropoint).
        """
        if self.has_light_profile:
            return sum(map(lambda p: p.luminosity_within_circle(radius, conversion_factor), self.light_profiles))
        else:
            return None

    def luminosity_within_ellipse(self, major_axis, conversion_factor=1.0):
        """Compute the total luminosity of the galaxy's light profiles, within an ellipse of specified major axis. This 
        is performed via integration of each light profile and is centred, oriented and aligned with each light
        model's individual geometry.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the photometric zeropoint).

        See *light_profiles.luminosity_within_ellipse* for details of how this is performed.

        Parameters
        ----------
        major_axis: float
            The major-axis of the ellipse to compute the luminosity within.
        conversion_factor : float
            Factor the dimensionless luminosity is multiplied by to convert it to a physical luminosity \
            (e.g. a photometric zeropoint).
        """
        if self.has_light_profile:
            return sum(map(lambda p: p.luminosity_within_ellipse(major_axis, conversion_factor), self.light_profiles))
        else:
            return None

    def convergence_from_grid(self, grid):
        """Compute the summed convergence of the galaxy's mass profiles using a grid of Cartesian (y,x) \
        coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.
        
        See *profiles.mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(map(lambda p: p.convergence_from_grid(grid), self.mass_profiles))
        else:
            return np.zeros((grid.shape[0],))

    def potential_from_grid(self, grid):
        """Compute the summed gravitational potential of the galaxy's mass profiles using a grid of Cartesian (y,x) \
        coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See *profiles.mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(map(lambda p: p.potential_from_grid(grid), self.mass_profiles))
        else:
            return np.zeros((grid.shape[0],))

    def deflections_from_grid(self, grid):
        """Compute the summed (y,x) deflection angles of the galaxy's mass profiles using a grid of Cartesian (y,x) \
        coordinates.

        If the galaxy has no mass profiles, two grid of zeros are returned.

        See *profiles.mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(map(lambda p: p.deflections_from_grid(grid), self.mass_profiles))
        else:
            return np.full((grid.shape[0], 2), 0.0)

    def mass_within_circle_in_angular_units(self, radius):
        """Compute the total angular mass of the galaxy's mass profiles within a circle of specified radius.

        The value returned by this integral is in angular units, however a conversion factor can be specified to \
        convert it to a physical value (e.g. the critical surface mass density).

        See *profiles.mass_profiles.angular_mass_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        conversion_factor : float
            Factor the dimensionless mass is multiplied by to convert it to a physical mass (e.g. the critical surface \
            mass density).
        """
        if self.has_mass_profile:
            return sum(map(lambda p: p.mass_within_circle_in_angular_units(radius), self.mass_profiles))
        else:
            return None

    def mass_within_ellipse_in_angular_units(self, major_axis):
        """Compute the total angular mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        The value returned by this integral is in angular units, however a conversion factor can be specified to \
        convert it to a physical value (e.g. the critical surface mass density).

        See *profiles.mass_profiles.angualr_mass_within_ellipse* for details of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major axis of the ellipse
        conversion_factor : float
            Factor the dimensionless mass is multiplied by to convert it to a physical mass (e.g. the critical surface \
            mass density).
        """
        if self.has_mass_profile:
            return sum(map(lambda p: p.mass_within_ellipse_in_angular_units(major_axis), self.mass_profiles))
        else:
            return None

    def mass_within_circle_in_mass_units(self, radius, critical_surface_mass_density):
        """Compute the total angular mass of the galaxy's mass profiles within a circle of specified radius.

        The value returned by this integral is in angular units, however a conversion factor can be specified to \
        convert it to a physical value (e.g. the critical surface mass density).

        See *profiles.mass_profiles.angular_mass_within_circle* for details of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        conversion_factor : float
            Factor the dimensionless mass is multiplied by to convert it to a physical mass (e.g. the critical surface \
            mass density).
        """
        if self.has_mass_profile:
            return critical_surface_mass_density * self.mass_within_circle_in_angular_units(radius=radius)
        else:
            return None

    def mass_within_ellipse_in_mass_units(self, major_axis, critical_surface_mass_density):
        """Compute the total angular mass of the galaxy's mass profiles within an ellipse of specified major_axis.

        The value returned by this integral is in angular units, however a conversion factor can be specified to \
        convert it to a physical value (e.g. the critical surface mass density).

        See *profiles.mass_profiles.angualr_mass_within_ellipse* for details of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major axis of the ellipse
        conversion_factor : float
            Factor the dimensionless mass is multiplied by to convert it to a physical mass (e.g. the critical surface \
            mass density).
        """
        if self.has_mass_profile:
            return critical_surface_mass_density * self.mass_within_ellipse_in_angular_units(major_axis=major_axis)
        else:
            return None

    @property
    def einstein_radius(self):
        """The Einstein Radius of this galaxy, which is the sum of Einstein Radii of its mass profiles.

        If the galaxy is composed of multiple ellipitcal profiles with different axis-ratios, this Einstein Radius \
        may be inaccurate. This is because the differently oriented ellipses of each mass profile """

        if self.has_mass_profile:
            return sum(map(lambda p: p.einstein_radius, self.mass_profiles))
        else:
            return None


class HyperGalaxy(object):
    _ids = count()

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        """ If a *Galaxy* is given a *HyperGalaxy* as an attribute, the noise-map in the regions of the image that the \
        galaxy is located will be scaled, to prevent over-fitting of the galaxy. 
        
        This is performed by first computing the hyper-galaxy's 'contribution-map', which determines the fraction of \
        flux in every pixel of the image that can be associated with this particular hyper-galaxy. This is computed \
        using  hyper-hyper set (e.g. fitting.fit_data.FitDataHyper), which includes  best-fit unblurred_image_1d of \
        the galaxy's light from a previous analysis phase.
         
        The *HyperGalaxy* class contains the hyper-parameters which are associated with this galaxy for scaling the \
        noise-map.
        
        Parameters
        -----------
        contribution_factor : float
            Factor that adjusts how much of the galaxy's light is attributed to the contribution map.
        noise_factor : float
            Factor by which the noise-map is increased in the regions of the galaxy's contribution map.
        noise_power : float
            The power to which the contribution map is raised when scaling the noise-map.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

        self.component_number = next(self._ids)

    def hyper_noise_from_model_image_galaxy_image_and_noise_map(self, model_image, galaxy_image, noise_map,
                                                                minimum_value=0.0):
        contributions = self.contributions_from_model_image_and_galaxy_image(model_image, galaxy_image, minimum_value)
        return self.hyper_noise_from_contributions(noise_map, contributions)

    def contributions_from_model_image_and_galaxy_image(self, model_image, galaxy_image, minimum_value=0.0):
        """Compute the contribution map of a galaxy, which represents the fraction of flux in each pixel that the \
        galaxy is attributed to contain, scaled to the *contribution_factor* hyper-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that pixel, and then scaling by the \
        maximum flux such that the contribution map ranges between 0 and 1.

        Parameters
        -----------
        model_image : ndarray
            The best-fit model image to the observed image from a previous analysis phase. This provides the \
            total light attributed to each image pixel by the model.
        galaxy_image : ndarray
            A model image of the galaxy (from light profiles or an inversion) from a previous analysis phase.
        minimum_value : float
            The minimum contribution value a pixel must contain to not be rounded to 0.
        """
        contributions = np.divide(galaxy_image, np.add(model_image, self.contribution_factor))
        contributions = np.divide(contributions, np.max(contributions))
        contributions[contributions < minimum_value] = 0.0
        return contributions

    def hyper_noise_from_contributions(self, noise_map, contributions):
        """Compute a scaled galaxy hyper noise-map from a baseline noise-map.

        This uses the galaxy contribution map and the *noise_factor* and *noise_power* hyper-parameters.

        Parameters
        -----------
        noise_map : ndarray
            The observed noise-map (before scaling).
        contributions : ndarray
            The galaxy contribution map.
        """
        return self.noise_factor * (noise_map * contributions) ** self.noise_power

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

    @property
    def value(self):
        try:
            return self.redshift.value
        except AttributeError:
            return self.redshift

    def __str__(self):
        return str(self.redshift)
