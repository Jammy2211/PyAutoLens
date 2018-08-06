from autolens.pipeline import phase as ph
from autolens.autopipe import non_linear as nl
from autolens.analysis import galaxy_prior as gp
from autolens.pixelization import pixelization as px
from autolens.imaging import mask as msk
from autolens.profiles import light_profiles, mass_profiles

import logging

logger = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self, *phases):
        """

        Parameters
        ----------
        phases: [ph.Phase]
            Phases
        """
        self.phases = phases

    def run(self, image):
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.name, i))
            results.append(phase.run(image, ph.ResultsCollection(results)))
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
            A pipeline that runs all the  phases from this pipeline and then all the phases from the other pipeline
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


def make_profile_pipeline(name="profile_pipeline", optimizer_class=nl.MultiNest):
    # 1) Lens Light : EllipticalSersic
    #    Mass: None
    #    Source: None
    #    NLO: MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    phase1 = ph.LensOnlyPhase(lens_galaxy=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                              optimizer_class=optimizer_class, name="{}/phase1".format(name))

    class LensSubtractedPhase(ph.SourceLensPhase):
        def modify_image(self, masked_image, previous_results):
            return masked_image - previous_results.last.lens_galaxy_image

        def pass_priors(self, previous_results):
            self.lens_galaxy.sie.centre = previous_results.last.variable.lens_galaxy.elliptical_sersic.centre

    # 2) Lens Light : None
    #    Mass: SIE (use lens light profile centre from previous phase as prior on mass profile centre)
    #    Source: EllipticalSersic
    #    NLO: MultiNest
    #    Image : Lens Subtracted Image (previous phase)
    #    Mask : Annulus (0.4" - 3.0")

    def annular_mask_function(img):
        return msk.Mask.annular(img.shape_arc_seconds, pixel_scale=img.pixel_scale, inner_radius=0.4,
                                outer_radius=3.)

    phase2 = LensSubtractedPhase(lens_galaxy=gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermal),
                                 source_galaxy=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                                 optimizer_class=optimizer_class,
                                 mask_function=annular_mask_function,
                                 name="{}/phase2".format(name))

    # 3) Lens Light : Elliptical Sersic (Priors phase 1)
    #    Mass: SIE (Priors phase 2)
    #    Source : Elliptical Sersic (Priors phase 2)
    #    NLO : MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    class CombinedPhase(ph.SourceLensPhase):
        def pass_priors(self, previous_results):
            self.lens_galaxy = gp.GalaxyPrior(
                elliptical_sersic=previous_results.first.variable.lens_galaxy.elliptical_sersic,
                sie=previous_results.last.variable.lens_galaxy.sie)
            self.source_galaxy = previous_results.last.variable.source_galaxy

    phase3 = CombinedPhase(optimizer_class=optimizer_class,
                           name="{}/phase3".format(name))

    # 3H) Hyper-Parameters: Make Lens Galaxy and Source Galaxy Hyper-Galaxies.
    #     Lens Light / Mass / Source - Fix parameters to phase 3 most likely result
    #     NLO : DownhillSimplex
    #     Image : Observed Image
    #     Mask : Circle - 3.0"

    phase3h = ph.SourceLensHyperGalaxyPhase(name="{}/phase3h".format(name))

    # 4) Repeat phase 3, using its priors and the hyper-galaxies fixed to their optimized values.
    #    Lens Light : Elliptical Sersic (Priors phase 3)
    #    Mass: SIE (Priors phase 3)
    #    Source : Elliptical Sersic (Priors phase 3)
    #    NLO : MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    class CombinedPhase2(ph.SourceLensPhase):
        def pass_priors(self, previous_results):
            phase_3_results = previous_results[2]
            self.lens_galaxy = phase_3_results.variable.lens_galaxy
            self.source_galaxy = phase_3_results.variable.source_galaxy
            self.lens_galaxy.hyper_galaxy = previous_results.last.constant.lens_galaxy.hyper_galaxy
            self.source_galaxy.hyper_galaxy = previous_results.last.constant.source_galaxy.hyper_galaxy

    phase4 = CombinedPhase2(optimizer_class=optimizer_class, name="{}/phase4".format(name))

    return Pipeline(phase1, phase2, phase3, phase3h, phase4)
