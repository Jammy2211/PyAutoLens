from autolens.profiles import light_profiles, mass_profiles


class EllipticalSersicMassAndLightProfile(light_profiles.EllipticalSersicLightProfile,
                                          mass_profiles.EllipticalSersicMass):
    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0):
        super().__init__(centre, axis_ratio, phi, intensity, effective_radius, sersic_index)
        super(mass_profiles.EllipticalSersicMass, self).__init__(centre, axis_ratio, phi, intensity, effective_radius,
                                                                 sersic_index)
