"""
Initialize the lens model of a 2-plane lens+source system.

This pipeline is composed of 3 phases:

1) Fit and subtract the lens light using an elliptical Sersic.
2) Fit the source using the lens subtracted image from phase 1, with an SIE mass model (lens) and elliptical
Sersic (source).
3) Fit the lens and source simulataneously, using priors based on the results of phases 1 and 2.
"""

name = "initializer"


def make():
    from autolens.pipeline import phase as ph
    from autolens.pipeline import pipeline as pl
    from autolens.autopipe import non_linear as nl
    from autolens.analysis import galaxy_prior as gp
    from autolens.imaging import mask as msk
    from autolens.profiles import light_profiles as lp
    from autolens.profiles import mass_profiles as mp

    phase1 = ph.LensProfilePhase(lens_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                 optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(name))

    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.8

    class LensSubtractedPhase(ph.LensMassAndSourceProfilePhase):

        def modify_image(self, masked_image, previous_results):
            return previous_results.last.lens_subtracted_image

        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sie.centre = previous_results.last.variable.lens_galaxies[0].sersic.centre

    def annular_mask_function(img):
        return msk.Mask.annular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, inner_radius=0.4, outer_radius=3.)

    phase2 = LensSubtractedPhase(lens_galaxies=[gp.GalaxyPrior(sie=mp.SphericalIsothermalMP)],
                                 source_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                 optimizer_class=nl.MultiNest, mask_function=annular_mask_function,
                                 phase_name="{}/phase2".format(name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    class LensSourcePhase(ph.LensMassAndSourceProfilePhase):

        def pass_priors(self, previous_results):
            self.lens_galaxies[0] = gp.GalaxyPrior(
                sersic=previous_results.first.variable.lens_galaxies[0].sersic,
                sie=previous_results.last.variable.lens_galaxies[0].sie)
            self.source_galaxies = previous_results.last.variable.source_galaxies

    phase3 = LensSourcePhase(optimizer_class=nl.MultiNest, phase_name="{}/phase3".format(name))
    
    phase3.optimizer.n_live_points = 60
    phase3.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging("profile_pipeline", phase1, phase2, phase3)
