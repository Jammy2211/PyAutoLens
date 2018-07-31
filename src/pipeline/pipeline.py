from src.pipeline import phase as ph
from src.autopipe import non_linear as nl
from src.analysis import galaxy_prior as gp
from src.pixelization import pixelization as px
from src.imaging import mask as msk
from src.profiles import light_profiles, mass_profiles

import logging

logger = logging.getLogger(__name__)


class Results(list):
    def __init__(self, results):
        super().__init__(results)

    @property
    def last(self):
        if len(self) > 0:
            return self[-1]
        return None


class Pipeline(object):
    def __init__(self, *phases):
        self.phases = phases

    def run(self, image):
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.__class__.__name__, i))
            results.append(phase.run(image, Results(results)))
        return results

    def __add__(self, other):
        """
        Compose two pipelines

        Parameters
        ----------
        other: Pipeline
            Another pipeline

        Returns
        -------
        composed_pipeline: Pipeline
            A pipeline that runs all the phases from this pipeline and then all the phases from the other pipeline
        """
        return Pipeline(*(self.phases + other.phases))


def make_source_only_pipeline():
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

    return Pipeline(phase1, phase2, phase3)


def make_profile_pipeline():
    # 1) Lens Light : EllipticalSersic
    #    Mass: None
    #    Source: None
    #    NLO: MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    phase1 = ph.LensOnlyPhase(lens_galaxy=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                              optimizer_class=nl.MultiNest)

    class LensSubtractedPhase(ph.SourceLensPhase):
        def customize_image(self, masked_image, previous_results):
            return masked_image - previous_results.last.lens_galaxy_image

        def pass_priors(self, previous_results):
            #  TODO: does this work?
            self.lens_galaxy.sie.centre = previous_results.variable.lens_galaxy.elliptical_sersic.centre

    # 2) Lens Light : None
    #    Mass: SIE (use lens light profile centre from previous phase as prior on mass profile centre)
    #    Source: EllipticalSersic
    #    NLO: MultiNest
    #    Image : Lens Subtracted Image (previous phase)
    #    Mask : Annulus (0.4" - 3.0")

    def mask_function(img):
        return msk.Mask.annular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, inner_radius=0.4,
                                outer_radius=3.)

    phase2 = LensSubtractedPhase(lens_galaxy=gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermal),
                                 source_galaxy=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                                 optimizer_class=nl.MultiNest,
                                 mask_function=mask_function)

    return Pipeline(phase1, phase2)
