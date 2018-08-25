from autolens.profiles import light_profiles, mass_profiles

"""
Mass and light profiles describe both mass distributions and light distributions with a single set of parameters. This
means that the light and mass of these profiles are tied together. Galaxy and GalaxyPrior instances interpret these
objects as being both mass and light profiles. 
"""


class EllipticalSersicLMP(light_profiles.EllipticalSersicLP, mass_profiles.EllipticalSersicMP):
    
    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, mass_to_light_ratio=1.0):
        
        light_profiles.EllipticalSersicLP.__init__(self, centre, axis_ratio, phi, intensity, effective_radius,
                                                   sersic_index)
        mass_profiles.EllipticalSersicMP.__init__(self, centre, axis_ratio, phi, intensity, effective_radius,
                                                  sersic_index, mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n', r'\Psi']

class SphericalSersicLMP(EllipticalSersicLMP):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, sersic_index=4.0,
                 mass_to_light_ratio=1.0):
        """
        The SphericalSersicMP mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalSersicLMP.__init__(self, centre, 1.0, 0.0, intensity, effective_radius, sersic_index,
                                     mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'I', 'R', 'n', r'\Psi']

class EllipticalExponentialLMP(EllipticalSersicLMP):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        The EllipticalExponentialMP mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
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
        EllipticalSersicLMP.__init__(self, centre, axis_ratio, phi, intensity, effective_radius, 1.0,
                                     mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', r'\Psi']

class SphericalExponentialLMP(EllipticalExponentialLMP):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, mass_to_light_ratio=1.0):
        """
        The SphericalExponentialMP mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalExponentialLMP.__init__(self, centre, 1.0, 0.0, intensity, effective_radius, mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'I', 'R', r'\Psi']

class EllipticalDevVaucouleursLMP(EllipticalSersicLMP):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 mass_to_light_ratio=1.0):
        """
        The EllipticalDevVaucouleursMP mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
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
        super(EllipticalDevVaucouleursLMP, self).__init__(centre, axis_ratio, phi, intensity,
                                                          effective_radius, 4.0,
                                                          mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', r'\Psi']

class SphericalDevVaucouleursLMP(EllipticalDevVaucouleursLMP):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, mass_to_light_ratio=1.0):
        """
        The SphericalDevVaucouleursMP mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the centre of the profiles
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The radius containing half the light of this model_mapper
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        EllipticalDevVaucouleursLMP.__init__(self, centre, 1.0, 0.0, intensity, effective_radius, mass_to_light_ratio)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'I', 'R', r'\Psi']

class EllipticalSersicRadialGradientLMP(light_profiles.EllipticalSersicLP,
                                        mass_profiles.EllipticalSersicRadialGradientMP):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, mass_to_light_ratio=1.0, mass_to_light_gradient=0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The centre of the profiles
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
        light_profiles.EllipticalSersicLP.__init__(self, centre, axis_ratio, phi, intensity, effective_radius,
                                                   sersic_index)
        mass_profiles.EllipticalSersicRadialGradientMP.__init__(self, centre, axis_ratio, phi, intensity,
                                                                effective_radius,
                                                                sersic_index, mass_to_light_ratio,
                                                                mass_to_light_gradient)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'q', r'\phi', 'I', 'R', 'n', r'\Psi', r'\Gamma']

class SphericalSersicRadialGradientLMP(EllipticalSersicRadialGradientLMP):

    def __init__(self, centre=(0.0, 0.0), intensity=0.1, effective_radius=0.6, sersic_index=4.0,
                 mass_to_light_ratio=1.0, mass_to_light_gradient=0.0):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The centre of the profiles
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
        EllipticalSersicRadialGradientLMP.__init__(self, centre, 1.0, 0.0, intensity, effective_radius, sersic_index,
                                                   mass_to_light_ratio, mass_to_light_gradient)

    @property
    def parameter_labels(self):
        return ['x', 'y', 'I', 'R', 'n', r'\Psi', r'\Gamma']