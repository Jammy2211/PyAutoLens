from src.pipeline import phase as ph
from src.autopipe import non_linear as nl
from src.analysis import galaxy_prior as gp
from src.imaging import mask as msk
from src.profiles import light_profiles, mass_profiles


class Pipeline(object):
    def __init__(self, *phases):
        self.phases = phases
        self.results = []

    @property
    def last_result(self):
        return None if len(self.results) == 0 else self.results[-1]

    def run(self, image):
        for phase in self.phases:
            self.results.append(phase.run(image, self.last_result))


# 1) Mass: SIE+Shear
#    Source: Sersic
#    NLO: LM


def make_source_only_pipeline():
    phase1 = ph.SourceLensPhase(
        lens_galaxy=gp.GalaxyPrior(
            sie=mass_profiles.SphericalIsothermal,
            shear=mass_profiles.ExternalShear),
        source_galaxy=gp.GalaxyPrior(
            sersic=light_profiles.EllipticalSersic),
        optimizer_class=nl.DownhillSimplex,
        mask_function=lambda img: msk.Mask.circular(img.shape_arc_seconds, img.pixel_scale, 3))

    return Pipeline(phase1)
