from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline

# This pipeline fits a strong lens which has two lens galaxies, and it is composed of the following 4 phases:

# Phase 1) Fit the light profile of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").

# Phase 2) Fit the light profile of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").

# Phase 3) Use this lens-subtracted image to fit the source galaxy's light. The mass-profiles of the two lens galaxies
#          can use the results of phases 1 and 2 to initialize their priors.

# Phase 4) Fit all relevant parameters simultaneously, using priors from phases 1, 2 and 3.

# Because the pipeline assumes the lens galaxies are at (0.0", -1.0") and (0.0", 1.0"), it is not a general pipeline
# and cannot be applied to any image of a strong lens.

def make_pipeline(pipeline_path=''):

    pipeline_name = 'pipeline_x2_lens_galaxies'
    pipeline_path = pipeline_path + pipeline_name

    ### PHASE 1 ###

    # The left-hand galaxy is at (0.0", -1.0"), so we're going to use a small circular mask centred on its location to
    # fit its light profile. Its important that light from the other lens galaxy and source galaxy don't contaminate
    # our fit.

    def mask_function(img):
        return msk.Mask.circular(img.shape, pixel_scale=img.pixel_scale, radius_arcsec=0.5, centre=(0.0, -1.0))

    class LeftLensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            # Lets restrict the prior's on the centres around the pixel we know the galaxy's light centre peaks.

            self.lens_galaxies.left_lens.light.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.05)
            self.lens_galaxies.left_lens.light.centre_1 = mm.GaussianPrior(mean=-1.0, sigma=0.05)

            # Given we are only fitting the very central region of the lens galaxy, we don't want to let a parameter 
            # like th Sersic index vary. Lets fix it to 4.0.

            self.lens_galaxies.left_lens.light.sersic_index = 4.0

    phase1 = LeftLensPhase(lens_galaxies=dict(left_lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest, mask_function=mask_function,
                           phase_name=pipeline_path + '/phase_1_left_lens_light')

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # Now do the exact same with the lens galaxy on the right at (0.0", 1.0")

    def mask_function(img):
        return msk.Mask.circular(img.shape, pixel_scale=img.pixel_scale, radius_arcsec=0.5, centre=(0.0, 1.0))

    class RightLensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.right_lens.light.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.05)
            self.lens_galaxies.right_lens.light.centre_1 = mm.GaussianPrior(mean=1.0, sigma=0.05)
            self.lens_galaxies.right_lens.light.sersic_index = 4.0

    phase2 = RightLensPhase(lens_galaxies=dict(right_lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                            optimizer_class=nl.MultiNest, mask_function=mask_function,
                            phase_name=pipeline_path + '/phase_2_right_lens_light')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.5

    ### PHASE 3 ###

    # In the next phase, we fit the source of the lens subtracted image.

    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        # To modify the image, we want to subtract both the left-hand and right-hand lens galaxies. To do this, we need
        # to subtract the unmasked model image of both galaxies!

        def modify_image(self, image, previous_results):

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            return image - phase_1_results.unmasked_lens_plane_model_image - \
                   phase_2_results.unmasked_lens_plane_model_image

        def pass_priors(self, previous_results):

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]

            # We're going to link the centres of the light profiles computed above to the centre of the lens galaxy
            # mass-profiles in this phase. Because the centres of the mass profiles were fixed in phases 1 and 2,
            # linking them using the 'variable' attribute means that they stay constant (which for now, is what we want).

            self.lens_galaxies.left_lens.mass.centre_0 = phase_1_results.variable.left_lens.light.centre_0
            self.lens_galaxies.left_lens.mass.centre_1 = phase_1_results.variable.left_lens.light.centre_1

            self.lens_galaxies.right_lens.mass.centre_0 = phase_2_results.variable.right_lens.light.centre_0
            self.lens_galaxies.right_lens.mass.centre_1 = phase_2_results.variable.right_lens.light.centre_1

    phase3 = LensSubtractedPhase(lens_galaxies=dict(left_lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal),
                                                    right_lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                                 optimizer_class=nl.MultiNest,
                                 phase_name=pipeline_path + '/phase_3_fit_sources')

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    ### PHASE 4 ###

    # In phase 4, we'll fit both lens galaxy's light and mass profiles, as well as the source-galaxy, simultaneously.

    class FitAllPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            phase_3_results = previous_results[2]

            # Results are split over multiple phases, so we setup the light and mass profiles of each lens separately.

            self.lens_galaxies.left_lens.light = phase_1_results.variable.left_lens.light
            self.lens_galaxies.left_lens.mass = phase_3_results.variable.left_lens.mass
            self.lens_galaxies.right_lens.light = phase_2_results.variable.right_lens.light
            self.lens_galaxies.right_lens.mass = phase_3_results.variable.right_lens.mass

            # When we pass a a 'variable' galaxy from a previous phase, parameters fixed to constants remain constant.
            # Because centre_0 and centre_1 of the mass profile were fixed to constants in phase 3, they're still
            # constants after the line after. We need to therefore manually over-ride their priors.

            self.lens_galaxies.left_lens.mass.centre_0 = phase_3_results.variable.left_lens.mass.centre_0
            self.lens_galaxies.left_lens.mass.centre_1 = phase_3_results.variable.left_lens.mass.centre_1
            self.lens_galaxies.right_lens.mass.centre_0 = phase_3_results.variable.right_lens.mass.centre_0
            self.lens_galaxies.right_lens.mass.centre_1 = phase_3_results.variable.right_lens.mass.centre_1

            # We also want the Sersic index's to be free parameters now, so lets change it from a constant to a
            # variable.

            self.lens_galaxies.left_lens.light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=2.0)
            self.lens_galaxies.right_lens.light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=2.0)

            # Things are much simpler for the source galaxies - just like them togerther!

            self.source_galaxies.source = phase_3_results.variable.source

    phase4 = FitAllPhase(lens_galaxies=dict(left_lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                      mass=mp.EllipticalIsothermal),
                                            right_lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                        mass=mp.EllipticalIsothermal)),
                         source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                         optimizer_class=nl.MultiNest,
                         phase_name=pipeline_path + '/phase_4_fit_all')

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 60
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_path, phase1, phase2, phase3, phase4)
