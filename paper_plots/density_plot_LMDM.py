import sys
import os
import re
import getdist
import math
import numpy as np
from astropy import cosmology
sys.path.append("../")
sys.path.append("../autolens/")

from paper_plots import density_plot_tools

ltm_skip = 0
center_skip = 0

image_dir = '/gpfs/weighted_data/pdtw24/PL_Data/SL03_2/'  # Dir of Object to make evidence tables from

image_name = 'SLACSJ0252+0039'
image_name = 'SLACSJ1250+0523'
# image_name = 'SLACSJ1430+4105'

slacs = density_plot_tools.SLACS(image_dir, image_name)

pipeline_folder = 'Pipeline_LMDM'
phase_folder = 'PL_Phase_6'

# if image_name == 'SLACSJ1250+0523' or image_name == 'SLACSJ1430+4105':
#    center_skip = 2

image_dir = image_dir + image_name + '/' + pipeline_folder + '/' + phase_folder + '/'

model_folder = density_plot_tools.getModelFolders(image_dir)

pdf_file = image_dir + model_folder[0] + '/' + image_name + '.txt'

slacs.load_samples(pdf_file, center_skip, ltm_skip)

model_indexes, sample_weights, total_masses, stellar_masses, dark_masses, stellar_fractions = \
     slacs.masses_of_all_samples(radius_kpc=10.0)

total_mass, total_mass_std = density_plot_tools.weighted_avg_and_std(total_masses, sample_weights)
total_mass_lower = total_mass - total_mass_std
total_mass_upper = total_mass + total_mass_std

stellar_mass, stellar_mass_std = density_plot_tools.weighted_avg_and_std(stellar_masses, sample_weights)
stellar_mass_lower = stellar_mass - stellar_mass_std
stellar_mass_upper = stellar_mass + stellar_mass_std

slacs.weighted_densities_vs_radii(radius_kpc=30.0, weight_cut=1e-4, number_bins=50)

slacs.plot_density(image_name=image_name, labels=['Sersic Bulge', 'EllipticalExponentialMass Stellar Halo', 'Dark Matter Halo'])