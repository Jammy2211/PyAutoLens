from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.pipeline import phase as ph
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters
from test.simulation import simulation_util

import os

# Get the relative path to the config files and output folder in our workspace.
test_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output papth
conf.instance = conf.Config(config_path=test_path + 'config', output_path=test_path + 'output')

# It is convinient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.
data_type = 'no_lens_source_smooth'
data_resolution = 'Euclid'

# Setup the size of the sub-grid and mask used for this precision analysis.
sub_grid_size = 2
inner_radius_arcsec = 0.0
outer_radius_arcsec = 3.0

# The pixel scale of the interpolation grid, where a smaller pixel scale gives a higher resolution grid and therefore
# more precise interpolation of the sub-grid deflection angles.
interp_pixel_scale = 0.2

ccd_data = simulation_util.load_test_ccd_data(data_type=data_type, data_resolution=data_resolution, psf_shape=(21, 21))

# The phase is passed the mask we setup below using the radii specified above.
mask = msk.Mask.circular_annular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale,
                                 inner_radius_arcsec=inner_radius_arcsec, outer_radius_arcsec=outer_radius_arcsec)

# Plot the CCD data and mask.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search (in this case, MultiNest).
phase = ph.LensSourcePlanePhase(phase_name='phase_interp',
                                phase_folders=[data_type, data_resolution + '_' + str(interp_pixel_scale)],
                                lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest, interp_pixel_scale=interp_pixel_scale)

phase.optimizer.const_efficiency_mode = True
phase.optimizer.n_live_points = 50
phase.optimizer.sampling_efficiency = 0.5

# We run the phase on the image, print the results and plot the fit.
result = phase.run(data=ccd_data, mask=mask)