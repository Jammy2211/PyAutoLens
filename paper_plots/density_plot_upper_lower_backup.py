import sys
import os
import re
import getdist
import numpy as np
from astropy import cosmology

sys.path.append("../autolens/")

import galaxy
from profiles import light_profiles, mass_profiles


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
einstein_radius = 1.02
source_light_min = 0.8
source_light_max = 1.3
slacs_ein_r_kpc = 4.18
slacs_mass = '{0:e}'.format(10**11.25)
slacs_radius = 4.4
#
# image_name = 'SLACSJ1250+0523'
# redshift = 0.2318
# source_redshift = 0.7953
# einstein_radius = 1.120
# source_light_min = 0.5
# source_light_max = 1.5
# slacs_ein_r_kpc = 4.18
# slacs_mass =  '{0:e}'.format(10**11.26)
# slacs_chab_stellar_mass =  '{0:e}'.format(10**11.53)
# slacs_chab_stellar_mass_lower =  '{0:e}'.format(10**11.46)
# slacs_chab_stellar_mass_upper =  '{0:e}'.format(10**11.60)
# slacs_sal_stellar_mass =  '{0:e}'.format(10**11.77)
# slacs_sal_stellar_mass_lower =  '{0:e}'.format(10**11.7)
# slacs_sal_stellar_mass_upper =  '{0:e}'.format(10**12.84)

image_name = 'SLACSJ1430+4105'
redshift = 0.2850
source_redshift = 0.5753
einstein_radius = 1.468
source_light_min = 0.2
source_light_max = 2.3
our_einr_arc = 1.4668492852
slacs_ein_r_kpc = 6.53
slacs_mass =  '{0:e}'.format(10**11.73)
slacs_chab_stellar_mass =  '{0:e}'.format(10**11.68)
slacs_chab_stellar_mass_lower =  '{0:e}'.format(10**11.56)
slacs_chab_stellar_mass_upper =  '{0:e}'.format(10**11.8)
slacs_sal_stellar_mass =  '{0:e}'.format(10**11.93)
slacs_sal_stellar_mass_lower =  '{0:e}'.format(10**11.82)
slacs_sal_stellar_mass_upper =  '{0:e}'.format(10**12.04)

chab_stellar_frac = 0.33
chab_stellar_frac_error = 0.09

sal_stellar_frac = 0.59
sal_stellar_frac_error = 0.16

# pipeline_folder = 'Pipeline_Total'
# phase_folder = 'PL_Phase_8'

# pipeline_folder = 'Pipeline_LMDM'
# phase_folder = 'PL_Phase_7'

pipeline_folder = 'Pipeline_LTM2'
phase_folder = 'PL_Phase_7'

image_dir = image_dir + image_name + '/' + pipeline_folder + '/' + phase_folder + '/'

model_folder = getModelFolders(image_dir)

pdf_file = image_dir + model_folder[0] + '/' + image_name + '.txt'

pdf = getdist.mcsamples.loadMCSamples(pdf_file)

values = pdf.getMeans()
params = pdf.getParams()

upper_conf = []
lower_conf = []
for i in range(len(values)):
    upper_conf.append(pdf.confidence(paramVec=i, limfrac=0.95, upper=True))
    lower_conf.append(pdf.confidence(paramVec=i, limfrac=0.95, upper=False))

if image_name == 'SLACSJ1250+0523' or image_name == 'SLACSJ1430+4105':
    center_skip = 2

if pipeline_folder == 'Pipeline_LTM2':
    ltm_skip = 1

sersic_bulge = mass_profiles.SersicMassProfile(centre=(values[0], values[1]), axis_ratio=values[5], phi=values[6], intensity=values[2], effective_radius=values[3],
                                               sersic_index=values[4], mass_to_light_ratio=values[12+center_skip])

# setattr(sersic_bulge, 'centre_lower_limit_3_sigma', (lower_conf[0], lower_conf[1]))
# setattr(sersic_bulge, 'axis_ratio_lower_limit_3_sigma', lower_conf[5])
# setattr(sersic_bulge, 'phi_lower_limit_3_sigma', lower_conf[6])
# setattr(sersic_bulge, 'intensity_lower_limit_3_sigma', lower_conf[2])
# setattr(sersic_bulge, 'effective_radius_lower_limit_3_sigma', lower_conf[3])
# setattr(sersic_bulge, 'sersic_index_lower_limit_3_sigma', lower_conf[4])
# setattr(sersic_bulge, 'mass_to_light_ratio_lower_limit_3_sigma', lower_conf[12+center_skip])
#
# setattr(sersic_bulge, 'centre_upper_limit_3_sigma', (upper_conf[0], upper_conf[1]))
# setattr(sersic_bulge, 'axis_ratio_upper_limit_3_sigma', upper_conf[5])
# setattr(sersic_bulge, 'phi_upper_limit_3_sigma', upper_conf[6])
# setattr(sersic_bulge, 'intensity_upper_limit_3_sigma', upper_conf[2])
# setattr(sersic_bulge, 'effective_radius_upper_limit_3_sigma', upper_conf[3])
# setattr(sersic_bulge, 'sersic_index_upper_limit_3_sigma', upper_conf[4])
# setattr(sersic_bulge, 'mass_to_light_ratio_upper_limit_3_sigma', upper_conf[12+center_skip])

exponential_halo = mass_profiles.ExponentialMassProfile(axis_ratio=values[9+center_skip], phi=values[10+center_skip], intensity=values[7+center_skip],
                                                        effective_radius=values[8+center_skip], mass_to_light_ratio=values[12+ltm_skip+center_skip])

# setattr(exponential_halo, 'centre_lower_limit_3_sigma', (lower_conf[9+center_skip], lower_conf[10+center_skip]))
# setattr(exponential_halo, 'axis_ratio_lower_limit_3_sigma', lower_conf[9+center_skip])
# setattr(exponential_halo, 'phi_lower_limit_3_sigma', lower_conf[10+center_skip])
# setattr(exponential_halo, 'intensity_lower_limit_3_sigma', lower_conf[7+center_skip])
# setattr(exponential_halo, 'effective_radius_lower_limit_3_sigma', lower_conf[8+center_skip])
# setattr(exponential_halo, 'mass_to_light_ratio_lower_limit_3_sigma', lower_conf[12+ltm_skip+center_skip])
#
# setattr(exponential_halo, 'centre_upper_limit_3_sigma', (upper_conf[9+center_skip], upper_conf[10+center_skip]))
# setattr(exponential_halo, 'axis_ratio_upper_limit_3_sigma', upper_conf[9+center_skip])
# setattr(exponential_halo, 'phi_upper_limit_3_sigma', upper_conf[10+center_skip])
# setattr(exponential_halo, 'intensity_upper_limit_3_sigma', upper_conf[7+center_skip])
# setattr(exponential_halo, 'effective_radius_upper_limit_3_sigma', upper_conf[8+center_skip])
# setattr(exponential_halo, 'mass_to_light_ratio_upper_limit_3_sigma', upper_conf[12+ltm_skip+center_skip])

dark_matter_halo = mass_profiles.SphericalNFWMassProfile(kappa_s=values[11+center_skip], scale_radius=20.0)

setattr(dark_matter_halo, 'kappa_s_lower_limit_3_sigma', lower_conf[11+center_skip])
setattr(dark_matter_halo, 'kappa_s_upper_limit_3_sigma', upper_conf[11+center_skip])

lens_galaxy=galaxy.Galaxy(redshift=redshift, mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])

lens_galaxy.einstein_radius = einstein_radius
lens_galaxy.source_light_min = source_light_min
lens_galaxy.source_light_max = source_light_max

source_galaxy = galaxy.Galaxy(redshift=source_redshift)

galaxy.LensingPlanes(galaxies=[lens_galaxy, source_galaxy], cosmological_model=cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7))

scale_r = 30*lens_galaxy.arcsec_per_kpc
dark_matter_halo.scale_radius = scale_r
lens_galaxy.mass_profiles[2].scale_radius = scale_r
setattr(dark_matter_halo, 'scale_r_lower_limit_3_sigma', scale_r)
setattr(dark_matter_halo, 'scale_r_upper_limit_3_sigma', scale_r)

radius = slacs_ein_r_kpc*lens_galaxy.arcsec_per_kpc

total_mass = lens_galaxy.mass_within_circles(radius=radius)
masses = lens_galaxy.mass_within_circles_individual(radius=radius)
stellar_mass = (masses[0] + masses[1])

# lens_galaxy.mass_profiles[0].centre = lens_galaxy.mass_profiles[0].centre_lower_limit_3_sigma
# lens_galaxy.mass_profiles[0].axis_ratio = lens_galaxy.mass_profiles[0].axis_ratio_lower_limit_3_sigma
# lens_galaxy.mass_profiles[0].phi = lens_galaxy.mass_profiles[0].phi_lower_limit_3_sigma
# lens_galaxy.mass_profiles[0].intensity = lens_galaxy.mass_profiles[0].intensity_lower_limit_3_sigma
# lens_galaxy.mass_profiles[0].effective_radius = lens_galaxy.mass_profiles[0].effective_radius_lower_limit_3_sigma
# lens_galaxy.mass_profiles[0].sersic_index = lens_galaxy.mass_profiles[0].sersic_index_lower_limit_3_sigma
# lens_galaxy.mass_profiles[0].mass_to_light_ratio = lens_galaxy.mass_profiles[0].mass_to_light_ratio_lower_limit_3_sigma
#
# lens_galaxy.mass_profiles[1].centre = lens_galaxy.mass_profiles[1].centre_lower_limit_3_sigma
# lens_galaxy.mass_profiles[1].axis_ratio = lens_galaxy.mass_profiles[1].axis_ratio_lower_limit_3_sigma
# lens_galaxy.mass_profiles[1].phi = lens_galaxy.mass_profiles[1].phi_lower_limit_3_sigma
# lens_galaxy.mass_profiles[1].intensity = lens_galaxy.mass_profiles[1].intensity_lower_limit_3_sigma
# lens_galaxy.mass_profiles[1].effective_radius = lens_galaxy.mass_profiles[1].effective_radius_lower_limit_3_sigma
# lens_galaxy.mass_profiles[1].mass_to_light_ratio = lens_galaxy.mass_profiles[1].mass_to_light_ratio_lower_limit_3_sigma
#
# lens_galaxy.mass_profiles[2].kappa_s = lens_galaxy.mass_profiles[2].kappa_s_lower_limit_3_sigma
#
# total_mass_lower = lens_galaxy.mass_within_circles(radius=radius)
# masses_lower = lens_galaxy.mass_within_circles_individual(radius=radius)
# stellar_mass_lower = (masses_lower[0] + masses_lower[1])
#
#
# lens_galaxy.mass_profiles[0].centre = lens_galaxy.mass_profiles[0].centre_upper_limit_3_sigma
# lens_galaxy.mass_profiles[0].axis_ratio = lens_galaxy.mass_profiles[0].axis_ratio_upper_limit_3_sigma
# lens_galaxy.mass_profiles[0].phi = lens_galaxy.mass_profiles[0].phi_upper_limit_3_sigma
# lens_galaxy.mass_profiles[0].intensity = lens_galaxy.mass_profiles[0].intensity_upper_limit_3_sigma
# lens_galaxy.mass_profiles[0].effective_radius = lens_galaxy.mass_profiles[0].effective_radius_upper_limit_3_sigma
# lens_galaxy.mass_profiles[0].sersic_index = lens_galaxy.mass_profiles[0].sersic_index_upper_limit_3_sigma
# lens_galaxy.mass_profiles[0].mass_to_light_ratio = lens_galaxy.mass_profiles[0].mass_to_light_ratio_upper_limit_3_sigma
#
# lens_galaxy.mass_profiles[1].centre = lens_galaxy.mass_profiles[1].centre_upper_limit_3_sigma
# lens_galaxy.mass_profiles[1].axis_ratio = lens_galaxy.mass_profiles[1].axis_ratio_upper_limit_3_sigma
# lens_galaxy.mass_profiles[1].phi = lens_galaxy.mass_profiles[1].phi_upper_limit_3_sigma
# lens_galaxy.mass_profiles[1].intensity = lens_galaxy.mass_profiles[1].intensity_upper_limit_3_sigma
# lens_galaxy.mass_profiles[1].effective_radius = lens_galaxy.mass_profiles[1].effective_radius_upper_limit_3_sigma
# lens_galaxy.mass_profiles[1].mass_to_light_ratio = lens_galaxy.mass_profiles[1].mass_to_light_ratio_upper_limit_3_sigma
#
# lens_galaxy.mass_profiles[2].kappa_s = lens_galaxy.mass_profiles[2].kappa_s_upper_limit_3_sigma
#
# total_mass_upper = lens_galaxy.mass_within_circles(radius=radius)
# masses_upper = lens_galaxy.mass_within_circles_individual(radius=radius)
# stellar_mass_upper = (masses_upper[0] + masses_upper[1])

print(masses)

print('einstein radius = ', radius)

masses = lens_galaxy.mass_within_circles_individual(radius=radius)

print()
print('Our Mass = ', '{0:e}'.format(total_mass))
# print('Our Mass = ', '{0:e}'.format(total_mass), '(', '{0:e}'.format(total_mass_upper), '{0:e}'.format(total_mass_lower), ')')
print('SLACS Mass = ', slacs_mass)
print('Our Mass / SLACS Mass = ' , total_mass / float(slacs_mass) )

print()
# print('Stellar Mass fraction in R_Ein = ', stellar_mass / total_mass, '(', stellar_mass_upper / total_mass, stellar_mass_lower / total_mass, ')' )
print('Stellar Mass fraction in R_Ein = ', stellar_mass / total_mass)
print('SLACS Chabrier Stellar Mass Fraction in R_Ein = ', chab_stellar_frac, ' +- ', chab_stellar_frac_error )
print('SLACS Salpeter Stellar Mass Fraction in R_Ein = ', sal_stellar_frac, ' +- ', sal_stellar_frac_error  )

print()
# zrint('Our Stellar Mass = ','{0:e}'.format((stellar_mass)), '(', '{0:e}'.format(stellar_mass_upper), '{0:e}'.format(stellar_mass_lower), ')' )
print('Our Stellar Mass = ','{0:e}'.format((stellar_mass)) )
print('SLACS Chabrier stellar mass = ', '{0:e}'.format(float(slacs_chab_stellar_mass)), '(', '{0:e}'.format(float(slacs_chab_stellar_mass_lower)), '{0:e}'.format(float(slacs_chab_stellar_mass_upper)), ')'  )
print('SLACS Salpeter stellar mass = ', '{0:e}'.format(float(slacs_sal_stellar_mass)), '(', '{0:e}'.format(float(slacs_sal_stellar_mass_lower)), '{0:e}'.format(float(slacs_sal_stellar_mass_upper)), ')'  )
print('Our Stellar Mass / SLACS Chabrier Stellar Mass = ', stellar_mass / float(slacs_chab_stellar_mass))
print('Our Stellar Mass / SLACS Salpeter Stellar Mass = ', stellar_mass / float(slacs_sal_stellar_mass))


stop

print(lens_galaxy.mass_profiles[0].einstein_radius*lens_galaxy.kpc_per_arcsec_proper.value)
print((lens_galaxy.mass_within_circles_individual(radius=lens_galaxy.mass_profiles[0].einstein_radius*lens_galaxy.kpc_per_arcsec_proper.value)))
print(lens_galaxy.mass_within_circles_individual(radius=4.18))
print(lens_galaxy.mass_within_circles_individual(radius=radius*100.0))


stop

lens_galaxy.plot_density_as_function_of_radius(maximum_radius=5.0, number_bins=300, image_name=image_name, plot_errors=True,
                                               labels=['Sersic Bulge', 'EllipticalExponentialMass Stellar Halo', 'Dark Matter Halo'])