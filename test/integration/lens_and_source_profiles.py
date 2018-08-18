from autolens.pipeline import pipeline as pl
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.autopipe import non_linear as nl
from autolens import conf
from test.integration import tools

import numpy as np
import shutil
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
output_path = '/gpfs/data/pdtw24/Lens/integration/'

def test_lens_and_source_profiles_pipeline():

    pipeline_name = "profile_pipeline"
    data_name = '/lowres_lens_and_source'

    try:
        shutil.rmtree(dirpath+'/data'+data_name)
    except FileNotFoundError:
        pass

    lens_light_profile = lp.EllipticalSersic(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, intensity=1.0,
                                             effective_radius=1.3, sersic_index=3.0)
    lens_mass_profile = mp.EllipticalIsothermal(centre=(0.01, 0.01), axis_ratio=0.8, phi=0.0, einstein_radius=2.0)
    source_light_profile = lp.EllipticalSersic(centre=(0., 0.), axis_ratio=0.9, phi=90.0, intensity=1.0,
                                               effective_radius=1.0, sersic_index=2.0)

    simulate_lowres_integration_image(data_name=data_name, lens_light_profile=lens_light_profile,
                                      lens_mass_profile=lens_mass_profile, source_light_profile=source_light_profile)

    conf.instance.output_path = output_path

    try:
        shutil.rmtree(output_path)
    except FileNotFoundError:
        pass

    pipeline = make_lens_and_source_profiles_pipeline(pipeline_name=pipeline_name)
    image = load_image(data_name=data_name, pixel_scale=0.2)

    results = pipeline.run(image=image)
    for result in results:
        print(result)

def make_lens_and_source_profiles_pipeline(pipeline_name):

    from autolens.pipeline import phase as ph
    from autolens.analysis import galaxy_prior as gp
    from autolens.profiles import light_profiles

    # 1) Lens Light : EllipticalSersic
    #    Mass: None
    #    Source: None
    #    NLO: MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    phase1 = ph.LensPlanePhase(lens_galaxies=[gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic)],
                               optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    class LensSubtractedPhase(ph.LensSourcePhase):
        def modify_image(self, masked_image, previous_results):
            return masked_image - previous_results.last.lens_galaxy_image

        def pass_priors(self, previous_results):
            self.lens_galaxies.sie.centre = previous_results.last.variable.lens_galaxy.elliptical_sersic.centre

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
                                 source_galaxies=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersic),
                                 optimizer_class=nl.MultiNest,
                                 mask_function=annular_mask_function,
                                 phase_name="{}/phase2".format(pipeline_name))

    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    # 3) Lens Light : Elliptical Sersic (Priors phase 1)
    #    Mass: SIE (Priors phase 2)
    #    Source : Elliptical Sersic (Priors phase 2)
    #    NLO : MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    class CombinedPhase(ph.LensSourcePhase):
        def pass_priors(self, previous_results):
            self.lens_galaxy = gp.GalaxyPrior(
                elliptical_sersic=previous_results.first.variable.lens_galaxy.elliptical_sersic,
                sie=previous_results.last.variable.lens_galaxy.sie)
            self.source_galaxies = previous_results.last.variable.source_galaxies

    phase3 = CombinedPhase(optimizer_class=nl.MultiNest,
                           phase_name="{}/phase3".format(pipeline_name))

    # 3H) Hyper-Parameters: Make Lens Galaxy and Source Galaxy Hyper-Galaxies.
    #     Lens Light / Mass / Source - Fix parameters to phase 3 most likely result
    #     NLO : DownhillSimplex
    #     Image : Observed Image
    #     Mask : Circle - 3.0"

    phase3h = ph.SourceLensHyperGalaxyPhase(phase_name="{}/phase3h".format(pipeline_name))

    # 4) Repeat phase 3, using its priors and the hyper-galaxies fixed to their optimized values.
    #    Lens Light : Elliptical Sersic (Priors phase 3)
    #    Mass: SIE (Priors phase 3)
    #    Source : Elliptical Sersic (Priors phase 3)
    #    NLO : MultiNest
    #    Image : Observed Image
    #    Mask : Circle - 3.0"

    class CombinedPhase2(ph.LensSourcePhase):
        def pass_priors(self, previous_results):
            phase_3_results = previous_results[2]
            self.lens_galaxy = phase_3_results.variable.lens_galaxy
            self.source_galaxies = phase_3_results.variable.source_galaxies
            self.lens_galaxy.hyper_galaxy = previous_results.last.constant.lens_galaxy.hyper_galaxy
            self.source_galaxies.hyper_galaxy = previous_results.last.constant.source_galaxies.hyper_galaxy

    phase4 = CombinedPhase2(optimizer_class=nl.MultiNest, phase_name="{}/phase4".format(pipeline_name))

    return pl.Pipeline(pipeline_name, phase1, phase2, phase3, phase3h, phase4)


if __name__ == "__main__":
    test_lens_and_source_profiles_pipeline()
