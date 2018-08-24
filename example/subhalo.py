
pipeline_name = 'subhalo'

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

    ## PHASE 1 -- SUBTRACT LENS LIGHT

    phase1 = ph.LensProfilePhase(lens_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                 optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    ## PHASE 3 -- Fit Source Galaxy with profile

    class LensMassAtLightPhase(ph.LensMassAndSourceProfilePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies[0].sie.centre = previous_results[0].variable.lens_galaxies[0].sersic.centre

    def modify_image(self, image, previous_results):
        return image - previous_results.last.lens_plane_blurred_image_plane_image

    def annular_mask_function(img):
        return msk.Mask.annular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, inner_radius=0.6, outer_radius=3.)

    phase2 = LensMassAtLightPhase(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP)],
                                  source_galaxies=[gp.GalaxyPrior(sersic=lp.EllipticalSersicLP)],
                                  optimizer_class=nl.MultiNest, phase_name="{}/phase2".format(pipeline_name))

    ## PHASE 3 -- Fit Source Galaxy with pixelization

    class SourcePix(ph.LensMassAndSourcePixelizationPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies = previous_results[-1].variable.lens_galaxies
            self.source_galaxies[0].pixelization.shape_0 = mm.UniformPrior(19.5, 20.5)
            self.source_galaxies[0].pixelization.shape_1 = mm.UniformPrior(19.5, 20.5)

    phase3 = SourcePix(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP)],
                        source_galaxies=[gp.GalaxyPrior(pixelization=pix.RectangularRegConst)],
                        optimizer_class=nl.MultiNest, phase_name="{}/phase3".format(pipeline_name))

    ## Phase 4 - Model with Subhalo

    class SourcePix(ph.LensMassAndSourcePixelizationPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxies = previous_results[-1].variable.lens_galaxies
            self.source_galaxies = previous_results[-1].variable.source_galaxies

    phase4 = SourcePix(lens_galaxies=[gp.GalaxyPrior(sie=mp.EllipticalIsothermalMP),
                                      gp.GalaxyPrior(subhalo=mp.SphericalNFWMP)],
                        source_galaxies=[gp.GalaxyPrior(pixelization=pix.RectangularRegConst)],
                        optimizer_class=nl.MultiNest, phase_name="{}/phase4".format(pipeline_name))


    return pl.PipelineImaging("source_pix_pipeline", phase1, phase2, phase3, phase4)