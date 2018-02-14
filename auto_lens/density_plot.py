import sys
print(sys.path)

from galaxy import *
from profiles.geometry_profiles import *
from profiles.light_profiles import *
from profiles.mass_profiles import *

light_profile=CombinedLightProfile(
    SersicLightProfile(axis_ratio=0.88, phi=38.3, intensity=0.313, effective_radius=0.17, sersic_index=3.016, centre=(0.004, 0.015)),
    ExponentialLightProfile(axis_ratio=0.803, phi=109.6, intensity=0.0704, effective_radius=0.813))

sersic1 = SersicMassProfile.from_sersic_light_profile(light_profile[0], mass_to_light_ratio=5.306)
sersic2 = SersicMassProfile.from_sersic_light_profile(light_profile[1], mass_to_light_ratio=5.306)

mass_profile=CombinedMassProfile(SphericalNFWMassProfile(kappa_s=0.0388, scale_radius=10.0, centre=(0.004, 0.015)),
                                                          sersic1, sersic2)

#lens_slacs_0252 = Galaxy(redshift=0.2803, light_profile, mass_profile)