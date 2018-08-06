import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")

from paper_plots import density_plot_tools
from autolens.analysis import galaxy
from autolens.profiles import mass_profiles


ltm_skip = 0
center_skip = 0

image_dir = '/gpfs/weighted_data/pdtw24/PL_Data/SL03_2/'  # Dir of Object to make evidence tables from

image_name = 'SLACSJ0252+0039'
image_name = 'SLACSJ1250+0523'
# image_name = 'SLACSJ1430+4105'

slacs = density_plot_tools.SLACS(image_dir, image_name)

# pipeline_folder = 'Pipeline_Total'
# phase_folder = 'PL_Phase_8'

# pipeline_folder = 'Pipeline_LMDM'
# phase_folder = 'PL_Phase_7'

pipeline_folder = 'Pipeline_LTM2'
phase_folder = 'PL_5_LMDM_LTM_R_MC'

sersic_bulge = mass_profiles.EllipticalSersicMass(centre=(0.02, 0.008), axis_ratio=0.87,
                                                  phi=71.1, intensity=0.09458,
                                                  effective_radius=1.0153,
                                                  sersic_index=4.57468,
                                                  mass_to_light_ratio=1.55265)

exponential_halo = mass_profiles.EllipticalSersicMassRadialGradient(centre=(0.043, 0.036),
    axis_ratio=0.7861, phi=161.34, intensity=0.02475, effective_radius=2.1389, mass_to_light_ratio=1.2556,
                                                                    mass_to_light_gradient=-0.2229)

dark_matter_halo = mass_profiles.SphericalNFW(kappa_s=0.1035, scale_radius=30.0 * slacs.arcsec_per_kpc)

lens_galaxy = galaxy.Galaxy(redshift=slacs.redshift, mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])
source_galaxy = galaxy.Galaxy(redshift=slacs.source_redshift)

number_bins = 30
radius_kpc = 30

radii = list(np.linspace(5e-3, radius_kpc * slacs.arcsec_per_kpc, number_bins + 1))

radii_plot = []
bulge_densities = []
halo_densities = []
dark_densities = []

for r in range(number_bins):

    annuli_area = (math.pi * radii[r + 1] ** 2 - math.pi * radii[r] ** 2)

    densities = ((lens_galaxy.dimensionless_mass_within_circles_individual(radii[r + 1]) -
                  lens_galaxy.dimensionless_mass_within_circles_individual(radii[r])) / annuli_area)

    bulge_densities.append(densities[0])
    halo_densities.append(densities[1])
    dark_densities.append(densities[2])

    radii_plot.append(((radii[r + 1] + radii[r]) / 2.0) * slacs.kpc_per_arcsec)

print(bulge_densities)

plt.semilogy(radii_plot, bulge_densities, color='r', label='Sersic Bulge')
plt.semilogy(radii_plot, halo_densities, color='g', label='EllipticalExponentialMass Halo')
plt.semilogy(radii_plot, dark_densities, color='k', label='Dark Matter Halo')

plt.legend()
plt.show()

#slacs.weighted_densities_vs_radii(radius_kpc=30.0, weight_cut=1e-3, number_bins=20)

#slacs.plot_density(image_name=image_name, labels=['Sersic', 'Exponential', 'NFWSph'])