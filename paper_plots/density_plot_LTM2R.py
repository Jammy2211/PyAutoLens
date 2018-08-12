import sys

sys.path.append("../")

from paper_plots import density_plot_tools


ltm_skip = 0
center_skip = 0

image_dir = '/gpfs/data_vector/pdtw24/PL_Data/SL03_2/'  # Dir of Object to make evidence tables from

image_name = 'SLACSJ0252+0039'
image_name = 'SLACSJ1250+0523'
# image_name = 'SLACSJ1430+4105'

slacs = density_plot_tools.SLACS(image_dir, image_name)

# pipeline_folder = 'Pipeline_Total'
# phase_folder = 'PL_Phase_8'

# pipeline_folder = 'Pipeline_LMDM'
# phase_folder = 'PL_Phase_7'

pipeline_folder = 'Pipeline_LTM2'
phase_folder = 'PL_Phase_7'

if image_name == 'SLACSJ1250+0523' or image_name == 'SLACSJ1430+4105':
    center_skip = 2

if pipeline_folder == 'Pipeline_LTM2':
    ltm_skip = 1

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

# print()
# print('Our Mass = ', '{0:e}'.format(total_mass), '(', '{0:e}'.format(total_mass_lower), '{0:e}'.format(total_mass_upper), ')')
# print('SLACS Mass in R_Ein= ', slacs.mass)
# print('Our Mass / SLACS Mass = ' , total_mass / float(slacs.mass) )
#
# print()
# print('Stellar Mass fraction in R_Ein = ', stellar_mass / total_mass, '(', stellar_mass_lower / total_mass, stellar_mass_upper / total_mass, ')' )
# print('SLACS Chabrier Stellar Mass Fraction in R_Ein = ', slacs.chab_stellar_frac, ' +- ', slacs.chab_stellar_frac_error )
# print('SLACS Salpeter Stellar Mass Fraction in R_Ein = ', slacs.sal_stellar_frac, ' +- ', slacs.sal_stellar_frac_error  )
#
# print()
# print('Our Stellar Mass = ','{0:e}'.format((stellar_mass)), '(', '{0:e}'.format(stellar_mass_lower), '{0:e}'.format(stellar_mass_upper), ')' )
# print('SLACS Chabrier stellar mass = ', '{0:e}'.format(float(slacs.chab_stellar_mass)), '(', '{0:e}'.format(float(slacs.chab_stellar_mass_lower)), '{0:e}'.format(float(slacs.chab_stellar_mass_upper)), ')'  )
# print('SLACS Salpeter stellar mass = ', '{0:e}'.format(float(slacs.sal_stellar_mass)), '(', '{0:e}'.format(float(slacs.sal_stellar_mass_lower)), '{0:e}'.format(float(slacs.sal_stellar_mass_upper)), ')'  )
# print('Our Stellar Mass / SLACS Chabrier Stellar Mass = ', stellar_mass / float(slacs.chab_stellar_mass))
# print('Our Stellar Mass / SLACS Salpeter Stellar Mass = ', stellar_mass / float(slacs.sal_stellar_mass))


slacs.weighted_densities_vs_radii(radius_kpc=30.0, weight_cut=1e-3, number_bins=20)

slacs.plot_density(image_name=image_name, labels=['Sersic', 'Exponential', 'NFWSph'])