from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

# In this pipeline, we'll perform a basic analysis which fits two source galaxies using a parametric light profile and
# a lens galaxy where its light is not present in the image, using two phases:

# Phase 1) Fit the lens galaxy's mass (SIE+Shear) and source galaxy's light using a single Sersic light profile.

# Phase 2) Fit the lens galaxy's mass (SIE+Shear) and source galaxy's light using two Sersic light profiles,
#          initializing the priors of the lens and source using the results of Phase 1.

def make_pipeline(pipeline_path=''):

    pipeline_name = 'pipeline_no_lens_light_and_x2_source_parametric'
    pipeline_path = pipeline_path + pipeline_name

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale,
                                         inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    class LensSourceX1Phase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass.centre_0 = prior.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.mass.centre_1 = prior.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                      shear=mp.ExternalShear)),
                               source_galaxies=dict(source_0=gm.GalaxyModel(light=lp.EllipticalSersic)),
                               mask_function=mask_function, optimizer_class=nl.MultiNest,
                               phase_name=pipeline_path + '/phase_1_x1_source')

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

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Initialize the priors on the lens galaxy using the results of phase 1.
    # 2) Initialize the priors on the first source galaxy using the results of phase 1.

    class LensSourceX2Phase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[0].variable.lens
            self.source_galaxies.source_0 = previous_results[0].variable.source_0

    phase2 = LensSourceX2Phase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                      shear=mp.ExternalShear)),
                               source_galaxies=dict(source_0=gm.GalaxyModel(light=lp.EllipticalSersic),
                                                      source_1=gm.GalaxyModel(light=lp.EllipticalSersic)),
                               optimizer_class=nl.MultiNest, mask_function=mask_function,
                               phase_name=pipeline_path + '/phase_2_x2_source')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_path, phase1, phase2)