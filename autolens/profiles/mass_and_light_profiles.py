from autolens.profiles import light_profiles, mass_profiles


class EllipticalSersicMassAndLight(light_profiles.EllipticalSersic, mass_profiles.EllipticalSersicMass):
    def __init__(self):
        pass
