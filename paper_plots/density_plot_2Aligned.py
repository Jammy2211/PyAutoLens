import sys
import os
import re
import getdist
import math
import numpy as np
from astropy import cosmology

sys.path.append("../autolens/")

import galaxy
from profiles import light_profiles, mass_profiles


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def getModelFolders(dir):
    regex = re.compile('.*Hyper')
    folders = [name for name in os.listdir(dir) if os.path.isdir(dir+name) and re.match(regex, name) == None]
    return folders

ltm_skip = 0
center_skip = 0

image_dir = '/gpfs/data_vector/pdtw24/PL_Data/SL03_2/'  # Dir of Object to make evidence tables from
#
# image_name = 'SLACSJ0252+0039'
# redshift = 0.2803
# source_redshift = 0.9818
# einstein_radius = 1.02
# source_light_min = 0.8
# source_light_max = 1.3
# slacs_ein_r_kpc = 4.18
# slacs_mass = '{0:e}'.format(10**11.25)
# slacs_radius = 4.4
# slacs_chab_stellar_mass =  '{0:e}'.format(10**11.21)
# slacs_chab_stellar_mass_lower =  '{0:e}'.format(10**11.08)
# slacs_chab_stellar_mass_upper =  '{0:e}'.format(10**11.34)
# slacs_sal_stellar_mass =  '{0:e}'.format(10**11.46)
# slacs_sal_stellar_mass_lower =  '{0:e}'.format(10**11.33)
# slacs_sal_stellar_mass_upper =  '{0:e}'.format(10**11.59)
#
image_name = 'SLACSJ1250+0523'
redshift = 0.2318
source_redshift = 0.7953
einstein_radius = 1.120
source_light_min = 0.5
source_light_max = 1.5
slacs_ein_r_kpc = 4.18
slacs_mass =  '{0:e}'.format(10**11.26)
slacs_chab_stellar_mass =  '{0:e}'.format(10**11.53)
slacs_chab_stellar_mass_lower =  '{0:e}'.format(10**11.46)
slacs_chab_stellar_mass_upper =  '{0:e}'.format(10**11.60)
slacs_sal_stellar_mass =  '{0:e}'.format(10**11.77)
slacs_sal_stellar_mass_lower =  '{0:e}'.format(10**11.7)
slacs_sal_stellar_mass_upper =  '{0:e}'.format(10**12.84)

# image_name = 'SLACSJ1430+4105'
# redshift = 0.2850
# source_redshift = 0.5753
# einstein_radius = 1.468
# source_light_min = 0.2
# source_light_max = 2.3
# our_einr_arc = 1.4668492852
# slacs_ein_r_kpc = 6.53
# slacs_mass =  '{0:e}'.format(10**11.73)
# slacs_chab_stellar_mass =  '{0:e}'.format(10**11.68)
# slacs_chab_stellar_mass_lower =  '{0:e}'.format(10**11.56)
# slacs_chab_stellar_mass_upper =  '{0:e}'.format(10**11.8)
# slacs_sal_stellar_mass =  '{0:e}'.format(10**11.93)
# slacs_sal_stellar_mass_lower =  '{0:e}'.format(10**11.82)
# slacs_sal_stellar_mass_upper =  '{0:e}'.format(10**12.04)

chab_stellar_frac = 0.33
chab_stellar_frac_error = 0.09

sal_stellar_frac = 0.59
sal_stellar_frac_error = 0.16

pipeline_folder = 'Pipeline_LMDM'
phase_folder = 'PL_Phase_6'

image_dir = image_dir + image_name + '/' + pipeline_folder + '/' + phase_folder + '/'

model_folder = 'SersicEll[XY_New][Rot_New]+ExpEll[XY_Prev][Rot_Prev]+NFWSph[XY_Fg][Rot_Off]+LTM[XY_Fg][Rot_Fg]+Shear'

pdf_file = image_dir + model_folder + '/' + image_name + '.txt'

pdf = getdist.mcsamples.loadMCSamples(pdf_file)
params = pdf.paramNames

sample_weights = []

total_mass_all = []
stellar_mass_all = []

for i in range(len(pdf.samples)):

    if pdf.weights[i] > 1e-6:

        values = pdf.samples[i]

    #    values = pdf.getMeans()

        sersic_bulge = mass_profiles.SersicMassProfile(centre=(values[0], values[1]), axis_ratio=values[5], phi=values[6], intensity=values[2], effective_radius=values[3],
                                                       sersic_index=values[4], mass_to_light_ratio=values[11])

        exponential_halo = mass_profiles.ExponentialMassProfile(axis_ratio=values[9], phi=values[6], intensity=values[7],
                                                                effective_radius=values[8], mass_to_light_ratio=values[11])

        dark_matter_halo = mass_profiles.SphericalNFWMassProfile(kappa_s=values[10], scale_radius=20.0)

        lens_galaxy=galaxy.Galaxy(redshift=redshift, mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])

        lens_galaxy.einstein_radius = einstein_radius
        lens_galaxy.source_light_min = source_light_min
        lens_galaxy.source_light_max = source_light_max

        source_galaxy = galaxy.Galaxy(redshift=source_redshift)

        galaxy.LensingPlanes(galaxies=[lens_galaxy, source_galaxy], cosmological_model=cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7))

        scale_r = 30*lens_galaxy.arcsec_per_kpc
        dark_matter_halo.scale_radius = scale_r
        lens_galaxy.mass_profiles[2].scale_radius = scale_r

        radius = slacs_ein_r_kpc*lens_galaxy.arcsec_per_kpc
        radius = 200.0*lens_galaxy.arcsec_per_kpc

        sample_weights.append(pdf.weights[i])

        total_mass_all.append(lens_galaxy.mass_within_circles(radius=radius))
        masses = lens_galaxy.mass_within_circles_individual(radius=radius)
        stellar_mass_all.append((masses[0] + masses[1]))

print(masses)

total_mass, total_mass_std = weighted_avg_and_std(total_mass_all, sample_weights)
total_mass_lower = total_mass - total_mass_std
total_mass_upper = total_mass + total_mass_std

stellar_mass, stellar_mass_std = weighted_avg_and_std(stellar_mass_all, sample_weights)
stellar_mass_lower = stellar_mass - stellar_mass_std
stellar_mass_upper = stellar_mass + stellar_mass_std

print()
print('Our Mass = ', '{0:e}'.format(total_mass), '(', '{0:e}'.format(total_mass_lower), '{0:e}'.format(total_mass_upper), ')')
print('SLACS Mass in R_Ein= ', slacs_mass)
print('Our Mass / SLACS Mass = ' , total_mass / float(slacs_mass) )

print()
print('Stellar Mass fraction in R_Ein = ', stellar_mass / total_mass, '(', stellar_mass_lower / total_mass, stellar_mass_upper / total_mass, ')' )
print('SLACS Chabrier Stellar Mass Fraction in R_Ein = ', chab_stellar_frac, ' +- ', chab_stellar_frac_error )
print('SLACS Salpeter Stellar Mass Fraction in R_Ein = ', sal_stellar_frac, ' +- ', sal_stellar_frac_error  )

print()
print('Our Stellar Mass = ','{0:e}'.format((stellar_mass)), '(', '{0:e}'.format(stellar_mass_lower), '{0:e}'.format(stellar_mass_upper), ')' )
print('SLACS Chabrier stellar mass = ', '{0:e}'.format(float(slacs_chab_stellar_mass)), '(', '{0:e}'.format(float(slacs_chab_stellar_mass_lower)), '{0:e}'.format(float(slacs_chab_stellar_mass_upper)), ')'  )
print('SLACS Salpeter stellar mass = ', '{0:e}'.format(float(slacs_sal_stellar_mass)), '(', '{0:e}'.format(float(slacs_sal_stellar_mass_lower)), '{0:e}'.format(float(slacs_sal_stellar_mass_upper)), ')'  )
print('Our Stellar Mass / SLACS Chabrier Stellar Mass = ', stellar_mass / float(slacs_chab_stellar_mass))
print('Our Stellar Mass / SLACS Salpeter Stellar Mass = ', stellar_mass / float(slacs_sal_stellar_mass))

lens_galaxy.plot_density_as_function_of_radius(maximum_radius=20.0*lens_galaxy.arcsec_per_kpc, number_bins=300, image_name=image_name, plot_errors=False,
                                               labels=['Sersic Bulge', 'EllipticalExponentialMass Stellar Halo', 'Dark Matter Halo'])