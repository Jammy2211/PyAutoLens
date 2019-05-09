from autolens import dimensions as dim
from autolens.model.profiles import light_profiles, mass_profiles

"""
Mass and light profiles describe both mass distributions and light distributions with a single set of parameters. This
means that the light and mass of these profiles are tied together. Galaxy and GalaxyModel instances interpret these
objects as being both mass and light profiles. 
"""


class EllipticalSersic(light_profiles.EllipticalSersic, mass_profiles.EllipticalSersic):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):

        light_profiles.EllipticalSersic.__init__(self, centre=centre, axis_ratio=axis_ratio, phi=phi,
                                                 intensity=intensity, effective_radius=effective_radius,
                                                 sersic_index=sersic_index)
        mass_profiles.EllipticalSersic.__init__(self, centre=centre, axis_ratio=axis_ratio, phi=phi,
                                                intensity=intensity, effective_radius=effective_radius,
                                                sersic_index=sersic_index, mass_to_light_ratio=mass_to_light_ratio)


class SphericalSersic(EllipticalSersic):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The SphericalSersic mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The regular_grid of the origin of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalSersic.__init__(self, centre=centre, axis_ratio=1.0, phi=0.0,
                                  intensity=intensity, effective_radius=effective_radius, sersic_index=sersic_index,
                                  mass_to_light_ratio=mass_to_light_ratio)


class EllipticalExponential(EllipticalSersic):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The EllipticalExponential mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The regular_grid of the origin of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalSersic.__init__(self, centre=centre, axis_ratio=axis_ratio, phi=phi,
                                  intensity=intensity, effective_radius=effective_radius,
                                  sersic_index=1.0, mass_to_light_ratio=mass_to_light_ratio)


class SphericalExponential(EllipticalExponential):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The SphericalExponential mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The regular_grid of the origin of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalExponential.__init__(self, centre=centre, axis_ratio=1.0, phi=0.0, intensity=intensity,
                                       effective_radius=effective_radius, mass_to_light_ratio=mass_to_light_ratio)


class EllipticalDevVaucouleurs(EllipticalSersic):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The EllipticalDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The regular_grid of the origin of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(EllipticalDevVaucouleurs, self).__init__(centre=centre, axis_ratio=axis_ratio, phi=phi,
                                                       intensity=intensity, effective_radius=effective_radius,
                                                       sersic_index=4.0, mass_to_light_ratio=mass_to_light_ratio)


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0):
        """
        The SphericalDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The regular_grid of the origin of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalDevVaucouleurs.__init__(self, centre=centre, axis_ratio=1.0, phi=0.0, intensity=intensity,
                                          effective_radius=effective_radius, mass_to_light_ratio=mass_to_light_ratio)


class EllipticalSersicRadialGradient(light_profiles.EllipticalSersic,
                                     mass_profiles.EllipticalSersicRadialGradient):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 axis_ratio: float = 1.0,
                 phi: float = 0.0,
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
                 mass_to_light_gradient: float = 0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The origin of the profiles
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        light_profiles.EllipticalSersic.__init__(self, centre=centre, axis_ratio=axis_ratio, phi=phi,
                                                 intensity=intensity, effective_radius=effective_radius,
                                                 sersic_index=sersic_index)
        mass_profiles.EllipticalSersicRadialGradient.__init__(self, centre=centre, axis_ratio=axis_ratio, phi=phi,
                                                              intensity=intensity,
                                                              effective_radius=effective_radius,
                                                              sersic_index=sersic_index,
                                                              mass_to_light_ratio=mass_to_light_ratio,
                                                              mass_to_light_gradient=mass_to_light_gradient)


class SphericalSersicRadialGradient(EllipticalSersicRadialGradient):

    def __init__(self,
                 centre: dim.Position = (0.0, 0.0),
                 intensity: dim.Luminosity = 0.1,
                 effective_radius: dim.Length = 0.6,
                 sersic_index: float = 0.6,
                 mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
                 mass_to_light_gradient: float = 0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The origin of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        EllipticalSersicRadialGradient.__init__(self, centre=centre, axis_ratio=1.0, phi=0.0, intensity=intensity,
                                                effective_radius=effective_radius, sersic_index=sersic_index,
                                                mass_to_light_ratio=mass_to_light_ratio,
                                                mass_to_light_gradient=mass_to_light_gradient)
