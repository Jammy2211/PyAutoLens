from src.pipeline import phase as ph
from src.autopipe import non_linear as nl
from src.analysis import galaxy_prior as gp
from src.pixelization import pixelization as px
from src.imaging import mask as msk
from src.profiles import light_profiles, mass_profiles

import logging

logger = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self, *phases):
        self.phases = phases
        self.results = []

    @property
    def last_result(self):
        return None if len(self.results) == 0 else self.results[-1]

    def run(self, image):
        self.results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.__class__.__name__, i))
            self.results.append(phase.run(image, self.last_result))
        return self.results


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
        def pass_priors(self, last_results):
            self.lens_galaxy = last_results.variable.lens_galaxy

    phase2 = PriorLensPhase(pixelization=px.RectangularRegConst,
                            mask_function=mask_function)

    class ConstantLensPhase(ph.PixelizedSourceLensPhase):
        def pass_priors(self, last_results):
            self.lens_galaxy = last_results.constant.lens_galaxy

    phase3 = ConstantLensPhase(pixelization=px.RectangularRegConst,
                               mask_function=mask_function)

    return Pipeline(phase1, phase2, phase3)
