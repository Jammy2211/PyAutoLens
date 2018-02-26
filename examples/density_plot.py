import sys
sys.path.append("../auto_lens/")

import galaxy
from profiles import light_profiles, mass_profiles

sersic_bulge = mass_profiles.SersicMassProfile(axis_ratio=0.88, phi=38.3, intensity=0.313, effective_radius=0.17,
                                               sersic_index=3.016, centre=(0.004, 0.015), mass_to_light_ratio=5.306)

exponential_halo = mass_profiles.ExponentialMassProfile(axis_ratio=0.803, phi=109.6, intensity=0.0704,
                                                        effective_radius=0.813, mass_to_light_ratio=5.306)

dark_matter_halo = mass_profiles.SphericalNFWMassProfile(kappa_s=0.0388839, scale_radius=10.0)

slacs0252_0039=galaxy.Galaxy(redshift=0.2803, mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])

slacs0252_0039.plot_density_as_function_of_radius(maximum_radius=5.0, number_bins=300,
                                                  labels=['Sersic Bulge', 'Exponential Stellar Halo', 'Dark Matter Halo'])

#print(slacs0252_0039.dimensionless_mass_within_circle_individual(radius=4.0))