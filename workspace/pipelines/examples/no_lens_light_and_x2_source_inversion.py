from autofit import conf
from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.data.plotters import ccd_plotters

import os

# In this pipeline, we'll perform a basic analysis which fits two source galaxies using a inversion light profile 
# followed by an inversion, where the lens galaxy's light is not present in the image, using two phases:

# Phase 1) Fit the lens galaxy's mass (SIE+Shear) and source galaxy's light using a single Sersic light profile.

# Phase 2) Fit the lens galaxy's mass (SIE+Shear) and source galaxy's light using an inversion, initializing the priors 
#          of the lens the results of Phase 1.

# The first phase of this pipeline is identical to phase 1 of the 'no_lens_light_x2_source_parametric.py' pipeline.

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# There is a x2 '/../../' because we are in the 'workspace/pipelines/examples' folder. If you write your own pipeline \
# in the 'workspace/pipelines' folder you should remove one '../', as shown below.
# path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output papth
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# It is convenient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.
lens_name = 'example_no_lens_light_and_x2_source'

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + lens_name + '/image.fits', pixel_scale=0.1,
                                       psf_path=path+'/data/'+lens_name+'/psf.fits',
                                       noise_map_path=path+'/data/'+lens_name+'/noise_map.fits')

# It is generally a good idea to plot the image before you run a pipeline, and make sure your mask is appropriately \
# sized. We'll use an annular mask in this example, which is plotted below.

mask = msk.Mask.circular_annular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale,
                                 inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask)

def make_no_lens_light_and_x2_source_inversion_pipeline(pipeline_name):

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function_annular(image):
        return msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale,
                                         inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    class LensSourceX1Phase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.mass.centre_1 = mm.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                    shear=mp.ExternalShear)),
                             source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                             mask_function=mask_function_annular, optimizer_class=nl.MultiNest,
                             phase_name=pipeline_name + '/phase_1_x1_source')

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout 'tutorial_7_multinest_black_magic' in
    # 'howtolens/chapter_2_lens_modeling'.

    # Fitting the lens galaxy and source galaxy from uninitialized priors often risks MultiNest getting stuck in a
    # local maxima, especially for the image in this example which actually has two source galaxies. Therefore, whilst
    # I will continue to use constant efficiency mode to ensure fast run time, I've upped the number of live points
    # and decreased the sampling efficiency from the usual values to ensure the non-linear search is robust.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we fit the lens's mass and source galaxy using an inversion, where we will:

    # 1) Initialize the priors on the lens galaxy using the results of phase 1.
    # 2) Assume default priors for all source inversion parameters.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass = previous_results[0].variable.lens.mass
            self.lens_galaxies.lens.shear = previous_results[0].variable.lens.shear

    phase2 = InversionPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                   mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            optimizer_class=nl.MultiNest, mask_function=mask_function_annular,
                            phase_name=pipeline_name + '/phase_2_inversion')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)


pipeline_no_lens_light_and_x2_source_inversion = \
    make_no_lens_light_and_x2_source_inversion_pipeline(pipeline_name='example/no_lens_light_and_x2_source_inversion')

pipeline_no_lens_light_and_x2_source_inversion.run(data=ccd_data)