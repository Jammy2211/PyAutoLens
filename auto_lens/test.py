import prior
from profile.mass_profile import *
from pprint import pprint

class MockConfig(object):
    def __init__(self, d=None):
        if d is not None:
            self.d = d
        else:
            self.d = {}

    def get(self, class_name, var_name):
        try:
            return self.d[class_name][var_name]
        except KeyError:
            return ["u", 0, 1]


sersic1 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                           sersic_index=4.0)

sersic2 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0, effective_radius=0.6,
                                           sersic_index=4.0, centre=(100, 0))

combined = light_profile.CombinedLightProfile(sersic1, sersic2)

print(combined[0].intensity)

model = prior.ClassMappingPriorCollection(MockConfig())

model.add_class('lens_galaxy_mass', CombinedMassProfile(EllipticalNFWMassProfile, EllipticalNFWMassProfile))
print(model.lens_galaxy_mass[0])
reconstruction = model.reconstruction_for_vector(([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
print(reconstruction.lens_galaxy_mass[0])
print(reconstruction.lens_galaxy_mass[0].scale_radius)
stop

#print((model.lens_galaxy_mass.intensity))

model.add_class('lens_galaxy_mass', EllipticalNFWMassProfile)
reconstruction = model.reconstruction_for_vector(([0.5, 0.5, 0.5, 0.5, 0.5]))
print(reconstruction.lens_galaxy_mass)
print(reconstruction.lens_galaxy_mass.scale_radius)

#reconstruction = model.reconstruction_for_vector(([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
#


# print(model.lens_galaxy_mass.scale_radius.upper_limit)
#
# print(len(model.prior_models))
# print(len(model.priors))