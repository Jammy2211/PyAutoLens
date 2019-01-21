from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.mapper import model_mapper as mm
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters

import os

# In this pipeline, we show how custom masks and positions (generated using the tools/mask_maker.py and
# tools/positions_maker.py files) can be input and used by a pipeline. We'll using a simple two phase pipeline:

# Phase 1) Fit the lens galaxy's light using an elliptical Sersic light profile.

# Phase 2) Use this lens subtracted image to fit the lens galaxy's mass (SIE+Shear) and source galaxy's light (Sersic).
#          This phase will use a custom mask and positions.

# The second phase of this pipeline loads a custom mask, which is used instead of the default mask function. A set of \
# positions are also loaded and use to restrict mass models to only those where the positions trace close to one another.

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# There is a x2 '/../../' because we are in the 'workspace/pipelines/examples' folder. If you write your own pipeline \
# in the 'workspace/pipelines' folder you should remove one '../', as shown below.
# path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output papth
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# It is convinient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.
lens_name = 'lens_light_and_x1_source'

# Load the CCD data of the lens
ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/example/' + lens_name + '/image.fits', pixel_scale=0.1,
                                       psf_path=path+'/data/example/'+lens_name+'/psf.fits',
                                       noise_map_path=path+'/data/example/'+lens_name+'/noise_map.fits')

# Load its mask from a mask.fits file generated using the tools/mask_maker.py file.
mask = msk.load_mask_from_fits(mask_path=path + '/data/example/' + lens_name + '/mask.fits', pixel_scale=0.1)

# Load its positions from a positions.fits file generated using the tools/positions_maker.py file.
positions = ccd.load_positions(positions_path=path + '/data/example/' + lens_name + '/positions.dat')

# Lets plot an image of the ccd data, mask and positions to make sure they are chosen correctly.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask, positions=positions)

def make_mask_and_positions_pipeline(pipeline_name):

    ### PHASE 1 ###

    # In phase 1, we will:

    # 1) Assume the image is centred on the lens galaxy.

    # In terms of the custom mask and positions, for this phase we will:

    # 1) Specify a mask_function which uses a circular mask, as the annular custom mask created for this lens is
    #    not appropriate for subtracting the lens galaxy's light.
    # 2) Don't specify anything about using positions, given this phase is a lens-only phase with no mass model.

    def mask_function(image):
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.5)

    class LensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.light.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.light.centre_1 = mm.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                       optimizer_class=nl.MultiNest, mask_function=mask_function,
                       phase_name=pipeline_name + '/phase_1_lens_light_only')

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will:

    # 1) Use a lens-subtracted image using a model lens galaxy image from phase 1.
    # 2) Initialize the priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
    #    its light profile in phase 1.

    # In terms of the custom mask and positions, for this phase we will:

    # 1) Not specify a mask function to the phase, meaning that by default the custom mask passed to the pipeline when
    #    we run it will be used instead.
    # 2) Specify for this phase to use the positions to resample the mass model. By default, use_positions=False
    #    and must explicitly be input as True for a phase to perform this resampling.

    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            return image - previous_results[-1].unmasked_lens_plane_model_image

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass.centre_0 = previous_results[0].variable.lens.light.centre_0
            self.lens_galaxies.lens.mass.centre_1 = previous_results[0].variable.lens.light.centre_1

    phase2 = LensSubtractedPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                        shear=mp.ExternalShear)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest, mask_function=mask_function, use_positions=True,
                                 phase_name=pipeline_name + '/phase_2_source_custom_mask_and_positions')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)


pipeline_mask_and_positions = make_mask_and_positions_pipeline(pipeline_name='example/mask_and_positions')

# Note that here we pass the mask and positions loaded above to run the pipeline.
pipeline_mask_and_positions.run(data=ccd_data, mask=mask, positions=positions)