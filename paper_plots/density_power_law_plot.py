import sys
import os
import re
import getdist
import numpy as np

sys.path.append("../autolens/")

import galaxy
from profiles import light_profiles, mass_profiles
from astropy import cosmology

def getModelFolders(dir):
    regex = re.compile('.*Hyper')
    folders = [name for name in os.listdir(dir) if os.path.isdir(dir+name) and re.match(regex, name) == None]
    return folders

ltm_skip = 0
center_skip = 0

image_dir = '/gpfs/weighted_data/pdtw24/PL_Data/SL03_2/'  # Dir of Object to make evidence tables from

image_name = 'SLACSJ0252+0039'
redshift = 0.2803
source_redshift = 0.9818
scale_r = 6.83924
einstein_radius = 1.02
source_light_min = 0.8
source_light_max = 1.3
slacs_mass = '{0:e}'.format(10**11.25)
slacs_radius = 4.4

image_name = 'SLACSJ1250+0523'
redshift = 0.2318
source_redshift = 0.7953
scale_r = 7.8660
kpc_to_arc = 3.8138
einstein_radius = 1.120
source_light_min = 0.5
source_light_max = 1.5
slacs_mass =  '{0:e}'.format(10**11.26)
slacs_radius = 4.18

# image_name = 'SLACSJ1430+4105'
# redshift = 0.2850
# source_redshift = 0.5753
# scale_r = 6.771
# kpc_to_arc = 4.4309
# einstein_radius = 1.468
# source_light_min = 0.2
# source_light_max = 2.3
# slacs_mass =  '{0:e}'.format(10**11.73)
# slacs_radius = 6.53

# pipeline_folder = 'Pipeline_Total'
# phase_folder = 'PL_Phase_8'

pipeline_folder = 'Pipeline_Total'
phase_folder = 'PL_Phase_8'

image_dir = image_dir + image_name + '/' + pipeline_folder + '/' + phase_folder + '/'

model_folder = getModelFolders(image_dir)

pdf_file = image_dir + model_folder[0] + '/' + image_name + '.txt'

pdf = getdist.mcsamples.loadMCSamples(pdf_file)

values = pdf.getMeans()
params = pdf.getParams()

upper_conf = []
lower_conf = []
for i in range(len(values)):
    upper_conf.append(pdf.confidence(paramVec=i, limfrac=0.999, upper=True))
    lower_conf.append(pdf.confidence(paramVec=i, limfrac=0.999, upper=False))

if image_name == 'SLACSJ1250+0523' or image_name == 'SLACSJ1430+4105':
    center_skip = 2

sersic_bulge = light_profiles.SersicLightProfile(centre=(values[0], values[1]), axis_ratio=values[5], phi=values[6], intensity=values[2], effective_radius=values[3],
                                               sersic_index=values[4])

setattr(sersic_bulge, 'centre_lower_limit_3_sigma', (lower_conf[0], lower_conf[1]))
setattr(sersic_bulge, 'axis_ratio_lower_limit_3_sigma', lower_conf[5])
setattr(sersic_bulge, 'phi_lower_limit_3_sigma', lower_conf[6])
setattr(sersic_bulge, 'intensity_lower_limit_3_sigma', lower_conf[2])
setattr(sersic_bulge, 'effective_radius_lower_limit_3_sigma', lower_conf[3])
setattr(sersic_bulge, 'sersic_index_lower_limit_3_sigma', lower_conf[4])

setattr(sersic_bulge, 'centre_upper_limit_3_sigma', (upper_conf[0], upper_conf[1]))
setattr(sersic_bulge, 'axis_ratio_upper_limit_3_sigma', upper_conf[5])
setattr(sersic_bulge, 'phi_upper_limit_3_sigma', upper_conf[6])
setattr(sersic_bulge, 'intensity_upper_limit_3_sigma', upper_conf[2])
setattr(sersic_bulge, 'effective_radius_upper_limit_3_sigma', upper_conf[3])
setattr(sersic_bulge, 'sersic_index_upper_limit_3_sigma', upper_conf[4])

exponential_halo = light_profiles.ExponentialLightProfile(axis_ratio=values[9+center_skip], phi=values[10+center_skip], intensity=values[7+center_skip],
                                                        effective_radius=values[8+center_skip])

setattr(exponential_halo, 'centre_lower_limit_3_sigma', (lower_conf[9+center_skip], lower_conf[10+center_skip]))
setattr(exponential_halo, 'axis_ratio_lower_limit_3_sigma', lower_conf[9+center_skip])
setattr(exponential_halo, 'phi_lower_limit_3_sigma', lower_conf[10+center_skip])
setattr(exponential_halo, 'intensity_lower_limit_3_sigma', lower_conf[7+center_skip])
setattr(exponential_halo, 'effective_radius_lower_limit_3_sigma', lower_conf[8+center_skip])
setattr(exponential_halo, 'centre_upper_limit_3_sigma', (upper_conf[9+center_skip], upper_conf[10+center_skip]))
setattr(exponential_halo, 'axis_ratio_upper_limit_3_sigma', upper_conf[9+center_skip])
setattr(exponential_halo, 'phi_upper_limit_3_sigma', upper_conf[10+center_skip])
setattr(exponential_halo, 'intensity_upper_limit_3_sigma', upper_conf[7+center_skip])
setattr(exponential_halo, 'effective_radius_upper_limit_3_sigma', upper_conf[8+center_skip])

power_law_mass = mass_profiles.EllipticalPowerLawMassProfile(centre=(values[11+center_skip], values[12+center_skip]),
                                                             einstein_radius=values[13+center_skip],
                                                             axis_ratio=values[14+center_skip], phi=values[15+center_skip],
                                                             slope=values[16+center_skip])

# power_law_mass = mass_profiles.EllipticalPowerLawMassProfile(centre=(values[10], values[11]),
#                                                              einstein_radius=values[12],
#                                                              axis_ratio=values[13], phi=values[14],
#                                                              slope=values[15])

setattr(power_law_mass, 'centre_lower_limit_3_sigma', (upper_conf[11+center_skip], upper_conf[12+center_skip]))
setattr(power_law_mass, 'einstein_radius_lower_limit_3_sigma', lower_conf[13+center_skip])
setattr(power_law_mass, 'axis_ratio_lower_limit_3_sigma', lower_conf[14+center_skip])
setattr(power_law_mass ,'phi_lower_limit_3_sigma', lower_conf[15+center_skip])
setattr(power_law_mass ,'slope_lower_limit_3_sigma', lower_conf[16+center_skip])

setattr(power_law_mass, 'centre_upper_limit_3_sigma', (upper_conf[11+center_skip], upper_conf[12+center_skip]))
setattr(power_law_mass, 'einstein_radius_upper_limit_3_sigma', upper_conf[13+center_skip])
setattr(power_law_mass, 'axis_ratio_upper_limit_3_sigma', upper_conf[14+center_skip])
setattr(power_law_mass ,'phi_upper_limit_3_sigma', upper_conf[15+center_skip])
setattr(power_law_mass ,'slope_upper_limit_3_sigma', upper_conf[16+center_skip])

lens_galaxy=galaxy.Galaxy(redshift=redshift, light_profiles=[sersic_bulge, exponential_halo],
                          mass_profiles=[power_law_mass])

source_galaxy = galaxy.Galaxy(redshift=source_redshift)

galaxy.LensingPlanes(galaxies=[lens_galaxy, source_galaxy], cosmological_model=cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7))

radius = lens_galaxy.mass_profiles[0].einstein_radius

total_mass = lens_galaxy.mass_within_circles(radius=radius)

lens_galaxy.mass_profiles[0].centre = lens_galaxy.mass_profiles[0].centre_lower_limit_3_sigma
lens_galaxy.mass_profiles[0].einstein_radius = lens_galaxy.mass_profiles[0].einstein_radius_lower_limit_3_sigma
lens_galaxy.mass_profiles[0].phi = lens_galaxy.mass_profiles[0].phi_lower_limit_3_sigma
lens_galaxy.mass_profiles[0].slope = lens_galaxy.mass_profiles[0].slope_lower_limit_3_sigma

total_mass_lower = lens_galaxy.mass_within_circles(radius=radius)

lens_galaxy.mass_profiles[0].centre = lens_galaxy.mass_profiles[0].centre_upper_limit_3_sigma
lens_galaxy.mass_profiles[0].einstein_radius = lens_galaxy.mass_profiles[0].einstein_radius_upper_limit_3_sigma
lens_galaxy.mass_profiles[0].phi = lens_galaxy.mass_profiles[0].phi_upper_limit_3_sigma
lens_galaxy.mass_profiles[0].slope = lens_galaxy.mass_profiles[0].slope_upper_limit_3_sigma

total_mass_upper = lens_galaxy.mass_within_circles(radius=radius)

# print(lens_galaxy.mass_profiles[0].centre)
# print(lens_galaxy.mass_profiles[0].einstein_radius*kpc_to_arc)
# print(lens_galaxy.mass_profiles[0].axis_ratio)
# print(lens_galaxy.mass_profiles[0].phi)
# print(lens_galaxy.mass_profiles[0].slope)
# stop

# print(lens_galaxy.critical_density)

# print(slacs_mass)
# print(slacs_radius)


# einstein_radius = slacs_radius*lens_galaxy.arcsec_per_kpc

print(einstein_radius)
print('Our Mass = ', '{0:e}'.format(total_mass), '(', '{0:e}'.format(total_mass_upper), '{0:e}'.format(total_mass_lower), ')')
print(float(slacs_mass) / total_mass )

stop

lens_galaxy.plot_density_as_function_of_radius(maximum_radius=5.0, number_bins=300, image_name=image_name, plot_errors=True,
                                               labels=['Sersic Bulge', 'EllipticalExponentialMass Stellar Halo', 'Dark Matter Halo'])