"""
Initialize the lens model of a 2-plane lens+source system.

This pipeline is composed of 3 phases:

1) Fit and subtract the lens light using an elliptical Sersic.
2) Fit the source using the lens subtracted image from phase 1, with an SIE mass model (lens) and elliptical
Sersic (source).
3) Fit the lens and source simultaneously, using priors based on the results of phases 1 and 2.
"""

pipeline_name = 'initializer'


def make():
    from autolens.pipeline import phase
    from autolens.pipeline import pipeline
    from autofit.core import non_linear as nl
    from autolens.imaging import mask
    from autolens.model.galaxy import galaxy_model as gp
    from autolens.model.profiles import light_profiles as lp
    from autolens.model.profiles import mass_profiles as mp

    phase1 = phase.LensPlanePhase(lens_galaxies=dict(lens_galaxy=gp.GalaxyModel(sersic=lp.EllipticalSersic)),
                                  optimizer_class=nl.MultiNest, phase_name='ph1_subtract_lens')

    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.8

    class LensSubtractedPhase(phase.LensSourcePlanePhase):

        def modify_image(self, masked_image, previous_results):
            return previous_results.last.lens_subtracted_image

        def pass_priors(self, previous_results):
            self.lens_galaxies.lens_galaxy.sie.centre_0 = previous_results.last. \
                variable.lens_galaxies.lens_galaxy.sersic.centre_0
            self.lens_galaxies.lens_galaxy.sie.centre_1 = previous_results.last. \
                variable.lens_galaxies.lens_galaxy.sersic.centre_1

    def annular_mask_function(img):
        return mask.Mask.annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.4,
                                 outer_radius_arcsec=3.)

    phase2 = LensSubtractedPhase(lens_galaxies=dict(lens_galaxy=gp.GalaxyModel(sie=mp.SphericalIsothermal)),
                                 source_galaxies=dict(source_galaxy=gp.GalaxyModel(sersic=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest, mask_function=annular_mask_function,
                                 phase_name='ph2_fit_source')

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.6

    class LensSourcePhase(phase.LensSourcePlanePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies.lens_galaxy = gp.GalaxyModel(
                sersic=previous_results.first.variable.lens_galaxies.lens_galaxy.sersic,
                sie=previous_results.last.variable.lens_galaxies.lens_galaxy.sie)
            self.source_galaxies = previous_results.last.variable.source_galaxies

    phase3 = LensSourcePhase(optimizer_class=nl.MultiNest, phase_name='ph3_fit_all')

    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)
