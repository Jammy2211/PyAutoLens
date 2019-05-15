from autolens.model.profiles import light_profiles as lp

from autolens.model.profiles.plotters import profile_plotters

sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=0.0, intensity=1.0, effective_radius=1.0,
                             sersic_index=4.0)

profile_plotters.plot_luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
    light_profile=sersic, minimum_radius=1.0e-4, maximum_radius=10.0)