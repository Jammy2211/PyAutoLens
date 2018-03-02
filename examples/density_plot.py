import sys
import os
import re
import getdist
sys.path.append("../auto_lens/")

import galaxy
from profiles import light_profiles, mass_profiles


def getModelFolders(dir):
    regex = re.compile('.*Hyper')
    folders = [name for name in os.listdir(dir) if os.path.isdir(dir+name) and re.match(regex, name) == None]
    return folders

ltm_skip = 0
center_skip = 0

image_dir = '/gpfs/data/pdtw24/PL_Data/SL03_2/'  # Dir of Object to make evidence tables from

image_name = 'SLACSJ0252+0039'
redshift = 0.2803
scale_r = 6.83924
kpc_to_arc = 4.3865

image_name = 'SLACSJ1250+0523'
redshift = 0.2318
scale_r = 7.8660
kpc_to_arc = 3.8138

image_name = 'SLACSJ1430+4105'
redshift = 0.2850
scale_r = 6.771
kpc_to_arc = 4.4309

# pipeline_folder = 'Pipeline_Total'
# phase_folder = 'PL_Phase_8'

# pipeline_folder = 'Pipeline_LMDM'


pipeline_folder = 'Pipeline_LTM2'
phase_folder = 'PL_Phase_7'

image_dir = image_dir + image_name + '/' + pipeline_folder + '/' + phase_folder + '/'

model_folder = getModelFolders(image_dir)

print(model_folder)

pdf_file = image_dir + model_folder[0] + '/' + image_name + '.txt'

pdf = getdist.mcsamples.loadMCSamples(pdf_file)

values = pdf.getMeans()

if image_name == 'SLACSJ1250+0523' or image_name == 'SLACSJ1430+4105':
    center_skip = 2

if pipeline_folder == 'Pipeline_LTM2':
    ltm_skip = 1

sersic_bulge = mass_profiles.SersicMassProfile(centre=(values[0], values[1]), axis_ratio=values[5], phi=values[6], intensity=values[2], effective_radius=values[3],
                                               sersic_index=values[4], mass_to_light_ratio=values[12+center_skip])

exponential_halo = mass_profiles.ExponentialMassProfile(axis_ratio=values[9+center_skip], phi=values[10+center_skip], intensity=values[7+center_skip],
                                                        effective_radius=values[8+center_skip], mass_to_light_ratio=values[12+ltm_skip+center_skip])

dark_matter_halo = mass_profiles.SphericalNFWMassProfile(kappa_s=values[11+center_skip], scale_radius=scale_r)

slacs0252_0039=galaxy.Galaxy(redshift=0.2803, mass_profiles=[sersic_bulge, exponential_halo, dark_matter_halo])

slacs0252_0039.plot_density_as_function_of_radius(maximum_radius=5.0, number_bins=300,
                                                  labels=['Sersic Bulge', 'Exponential Stellar Halo', 'Dark Matter Halo'])

#print(slacs0252_0039.dimensionless_mass_within_circle_individual(radius=4.0))