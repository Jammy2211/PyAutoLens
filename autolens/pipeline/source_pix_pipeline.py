"""
Analyse a lens source system using only profiles.

This pipeline fits the source light with an Elliptical Sersic profile, the lens light with an Elliptical Sersic
profile and the lens mass with a Spherical Isothermal profile.
"""

name = "source_pixelization"

def make():
    from autolens.pipeline import phase as ph
    from autolens.pipeline import pipeline as pl
    from autolens.autopipe import non_linear as nl
    from autolens.autopipe import model_mapper as mm
    from autolens.analysis import galaxy_prior as gp
    from autolens.imaging import mask as msk
    from autolens.profiles import light_profiles as lp
    from autolens.profiles import mass_profiles as mp
    from autolens.pixelization import pixelization as pix

    optimizer_class = nl.MultiNest

    def annular_mask_function(img):
        return msk.Mask.annular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, inner_radius=0.6,
                                outer_radius=3.)

    phase1 = ph.LensMassAndSourceProfilePhase(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP)],
                                              source_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                              optimizer_class=optimizer_class, mask_function=annular_mask_function,
                                              phase_name="{}/phase1".format(name))

    phase1.optimizer.n_live_points = 60
    phase1.optimizer.sampling_efficiency = 0.8

    class SourcePix(ph.LensMassAndSourcePixelizationPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies = previous_results[0].variable.lens_galaxies
            self.source_galaxies[0].pixelization.shape_0 = mm.UniformPrior(19.5, 20.5)
            self.source_galaxies[0].pixelization.shape_1 = mm.UniformPrior(19.5, 20.5)

    phase2 = SourcePix(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP)],
                        source_galaxies=[gp.GalaxyPrior(pixelization=pix.RectangularRegConst)],
                        optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(name))

    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.Pipeline("source_pix_pipeline", phase1, phase2)