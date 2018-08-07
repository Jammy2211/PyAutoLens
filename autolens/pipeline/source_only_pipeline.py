"""
Analyse only the source galaxy.

This pipeline fits the source galaxy with a rectangular pixelization. It assumes the foreground galaxy's mass profile
comprises Spherical Isothermal and External Shear components.
"""

name = "source"


def make():
    from autolens.pipeline import phase as ph
    from autolens.pipeline import pipeline as pl
    from autolens.analysis import galaxy_prior as gp
    from autolens.pixelization import pixelization as px
    from autolens.imaging import mask as msk
    from autolens.profiles import light_profiles, mass_profiles

    # 1) Mass: SIE+Shear
    #    Source: Sersic
    #    NLO: LM
    def mask_function(img):
        return msk.Mask.circular(img.shape_arc_seconds, img.pixel_scale, 2)

    phase1 = ph.SourceLensPhase(
        lens_galaxy=gp.GalaxyPrior(
            sie=mass_profiles.SphericalIsothermal,
            shear=mass_profiles.ExternalShear),
        source_galaxy=gp.GalaxyPrior(
            sersic=light_profiles.EllipticalSersic),
        mask_function=mask_function)

    # 2) Mass: SIE+Shear (priors from phase 1)
    #    Source: 'smooth' pixelization (include regularization parameter(s) in the model)
    #    NLO: LM
    class PriorLensPhase(ph.PixelizedSourceLensPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxy = previous_results.last.variable.lens_galaxy

    phase2 = PriorLensPhase(pixelization=px.RectangularRegConst,
                            mask_function=mask_function)

    class ConstantLensPhase(ph.PixelizedSourceLensPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxy = previous_results.last.constant.lens_galaxy

    phase3 = ConstantLensPhase(pixelization=px.RectangularRegConst,
                               mask_function=mask_function)
    return pl.Pipeline(name, phase1, phase2, phase3)
