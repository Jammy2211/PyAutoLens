from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.mapper import model_mapper as mm
from autolens.data.array import mask as msk
from autolens.data import ccd
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters

import os

# In this pipeline, we'll perform an advanced analysis which fits a lens galaxy with three surrounding line-of-sight \
# galaxies (which are at different redshifts and thus define a multi-plane stronog lens configuration). The source \
# galaxy will be modeled using a parametric light profile in the initial phases, but switch to an inversion in the \
# later phases.

# For efficiency, we will subtract the lens galaxy's light and line-of-sight galaxies light, and then only fit a
# foreground subtracted image. This means we can use an annular mask tailored to the source galaxy's light, which we
# have setup already using the 'tools/mask_maker.py' scripts.

# This leads to a 5 phase pipeline:

# Phase 1) Fit and subtract the light profile of the lens galaxy (elliptical Sersic) and each line-of-sight
#          galaxy (spherical Sersic).

# Phase 2) Use this lens subtracted image to fit the lens galaxy's mass (SIE) and source galaxy's light (Sersic),
#          thereby omitting multi-plane ray-tracing.

# Phase 3) Initialize the resolution and regularization coefficient of the inversion using the best-fit lens model from
#          phase 2.

# Phase 4) Refit the lens galaxy's light and mass models using an inversion (with priors from the above phases). The
#          mass profile of the 3 line-of-sight galaxies is also included (as SIS profiles), but a single lens plane
#          is assumed.

# Phase 5) Fit the lens galaxy, line-of-sight galaxies and source-galaxy using multi-plane ray-tracing, where the
#          redshift of each line-of-sight galaxy is included in the non-linear search as a free parameter. This phase
#          uses priors initialized from phase 4.

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# There is a x2 '/../../' because we are in the 'workspace/pipelines/examples' folder. If you write your own pipeline \
# in the 'workspace/pipelines' folder you should remove one '../', as shown below.
# path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output papth
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# It is convinient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.
lens_name = 'multi_plane'

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/example/' + lens_name + '/image.fits', pixel_scale=0.1,
                                       psf_path=path+'/data/example/'+lens_name+'/psf.fits',
                                       noise_map_path=path+'/data/example/'+lens_name+'/noise_map.fits')

# Load its mask from a mask.fits file generated using the tools/mask_maker.py file.
mask = msk.load_mask_from_fits(mask_path=path + '/data/' + lens_name + '/mask.fits', pixel_scale=0.1)

# Phase 5 risks the inversion falling into the systematic solutions where the mass of the lens model is too high,
# resulting in the inversion reconstructing the image as a demagnified version of itself. This is because the \
# line-of-sight halos each have a mass, which during the modeling can go to very high values.

# For this reason, we include positions (drawn using the 'tools/positions.py' script) to prevent these solutions from
# existing in parameter space.

positions = ccd.load_positions(positions_path=path + '/data/' + lens_name + '/positions.dat')

ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

def make_multi_plane_pipeline(pipeline_name):

    # In phase 1, we will:

    # 1) Subtract the light of the main lens galaxy (located at (0.0", 0.0")) and the light of each line-of-sight
    # galaxy (located at (4.0", 4.0"), (3.6", -5.3") and (-3.1", -2.4"))

    class LensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.light.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.light.centre_1 = mm.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.los0.light.centre_0 = mm.GaussianPrior(mean=4.0, sigma=0.1)
            self.lens_galaxies.los0.light.centre_1 = mm.GaussianPrior(mean=4.0, sigma=0.1)
            self.lens_galaxies.los1.light.centre_0 = mm.GaussianPrior(mean=3.6, sigma=0.1)
            self.lens_galaxies.los1.light.centre_1 = mm.GaussianPrior(mean=-5.3, sigma=0.1)
            self.lens_galaxies.los2.light.centre_0 = mm.GaussianPrior(mean=-3.1, sigma=0.1)
            self.lens_galaxies.los2.light.centre_1 = mm.GaussianPrior(mean=-2.4, sigma=0.1)

    phase1 = LensPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic),
                                           los_0=gm.GalaxyModel(light=lp.SphericalSersic),
                                           los_1=gm.GalaxyModel(light=lp.SphericalSersic),
                                           los_2=gm.GalaxyModel(light=lp.SphericalSersic)),
                       optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_1_light_subtraction')

    # Customize MultiNest so it runs fast
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.const_efficiency_mode = True

    # In phase 2, we will:

    # 1) Fit this foreground subtracted image, using an SIE+Shear mass model and Sersic source.
    # 2) Use the input positions to resample inaccurate mass models.

    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            return image - previous_results[0].unmasked_lens_plane_model_image

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.mass.centre_1 = mm.GaussianPrior(mean=0.0, sigma=0.1)

    phase2 = LensSubtractedPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 use_positions=True,
                                 optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_2_source_parametric')

    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2
    phase2.optimizer.const_efficiency_mode = True

    # In phase 3, we will:

    #  1) Fit the foreground subtracted image using a source-inversion instead of parametric source, using lens galaxy
    #     priors from phase 2.

    class InversionPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            return image - previous_results[0].unmasked_lens_plane_model_image

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[1].constant.lens

            self.source_galaxies.source.pixelization.shape_0 = mm.UniformPrior(lower_limit=20.0, upper_limit=45.0)
            self.source_galaxies.source.pixelization.shape_1 = mm.UniformPrior(lower_limit=20.0, upper_limit=45.0)

    phase3 = InversionPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                                        optimizer_class=nl.MultiNest,
                                        phase_name=pipeline_name + '/phase_3_inversion_init')

    # Customize MultiNest so it runs fast
    phase3.optimizer.n_live_points = 10
    phase3.optimizer.sampling_efficiency = 0.5
    phase3.optimizer.const_efficiency_mode = True

    # In phase 4, we will:

    #  1) Fit the foreground subtracted image using a source-inversion with parameters initialized from phase 3.
    #  2) Initialize the lens galaxy priors using the results of phase 2, and include each line-of-sight galaxy's mass
    #     contribution (keeping its centre fixed to the light profile centred inferred in phase 1). This will assume
    #     a single lens plane for all line-of-sight galaxies.
    #  3) Use the input positions to resample inaccurate mass models.

    class SingleLensPlanePhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            return image - previous_results[0].unmasked_lens_plane_model_image

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens.mass = previous_results[2].variable.lens.mass

            self.lens_galaxies.los_0.mass.centre_0 = previous_results[0].constant.los_0.light.centre_0
            self.lens_galaxies.los_0.mass.centre_1 = previous_results[0].constant.los_0.light.centre_1
            self.lens_galaxies.los_1.mass.centre_0 = previous_results[0].constant.los_1.light.centre_0
            self.lens_galaxies.los_1.mass.centre_1 = previous_results[0].constant.los_1.light.centre_1
            self.lens_galaxies.los_2.mass.centre_0 = previous_results[0].constant.los_2.light.centre_0
            self.lens_galaxies.los_2.mass.centre_1 = previous_results[0].constant.los_2.light.centre_1

            self.source_galaxies.source = previous_results[2].variable.source

    phase4 = SingleLensPlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal),
                                                     los_0=gm.GalaxyModel(mass=mp.SphericalIsothermal),
                                                     los_1=gm.GalaxyModel(mass=mp.SphericalIsothermal),
                                                     los_2=gm.GalaxyModel(mass=mp.SphericalIsothermal)),
                                  source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                             regularization=reg.Constant)),
                                  use_positions=True,
                                  optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_4_single_plane')

    # Customize MultiNest so it runs fast
    phase4.optimizer.n_live_points = 60
    phase4.optimizer.sampling_efficiency = 0.2
    phase4.optimizer.const_efficiency_mode = True


    # In phase 5, we will fit the foreground subtracted image using a multi-plane ray tracer. This means that the
    # redshift of each line-of-sight galaxy is included as a free parameter (we assume the lens and source redshifts
    # are known).

    class MultiPlanePhase(ph.MultiPlanePhase):

        def modify_image(self, image, previous_results):
            return image - previous_results[0].unmasked_lens_plane_model_image

        def pass_priors(self, previous_results):

            self.galaxies.lens = previous_results[3].variable.lens

            self.galaxies.los_0.mass = previous_results[3].variable.los_0.mass
            self.galaxies.los_1.mass = previous_results[3].variable.los_1.mass
            self.galaxies.los_2.mass = previous_results[3].variable.los_2.mass

            self.galaxies.source = previous_results[3].variable.source

    phase5 = MultiPlanePhase(galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal),
                             los_0=gm.GalaxyModel(mass=mp.SphericalIsothermal, variable_redshift=True),
                             los_1=gm.GalaxyModel(mass=mp.SphericalIsothermal, variable_redshift=True),
                             los_2=gm.GalaxyModel(mass=mp.SphericalIsothermal, variable_redshift=True),
                             source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                   regularization=reg.Constant)),
                             use_positions=True,
                             optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_5_multi_plane')

    # Customize MultiNest so it runs fast
    phase5.optimizer.n_live_points = 60
    phase5.optimizer.sampling_efficiency = 0.2
    phase5.optimizer.const_efficiency_mode = True

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4, phase5)


pipeline_multi_plane = make_multi_plane_pipeline(pipeline_name='example/mask_and_positions')
pipeline_multi_plane.run(data=ccd_data, mask=mask, positions=positions)