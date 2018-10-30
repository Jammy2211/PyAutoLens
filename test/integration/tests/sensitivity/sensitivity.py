import os

from autolens.autofit import non_linear as nl
from autolens.galaxy import galaxy, galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.dirname(dirpath)
output_path = '{}/../output/sensitivity'.format(dirpath)


def pipeline():
    pipeline_name = "sensitivity"
    data_name = '/sensitivity'

    tools.reset_paths(data_name, pipeline_name, output_path)

    lens_galaxy = galaxy.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6),
                                subhalo=mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=0.1))
    source_galaxy = galaxy.Galaxy(light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5,
                                                            sersic_index=1.0))

    try:
        tools.simulate_integration_image(data_name=data_name, pixel_scale=0.2, lens_galaxies=[lens_galaxy],
                                         source_galaxies=[source_galaxy], target_signal_to_noise=30.0)
    except OSError:
        pass

    pipeline = make_pipeline(pipeline_name=pipeline_name)
    image = tools.load_image(data_name=data_name, pixel_scale=0.1)

    results = pipeline.run(image=image)
    for result in results:
        print(result)


def make_pipeline(pipeline_name):

    class SensitivePhase(ph.SensitivityPhase):

        def pass_priors(self, previous_results):
            print(dir(self.lens_galaxies[0]))
            self.lens_galaxies[0].lens.mass.centre_0 = 0.0
            self.lens_galaxies[0].lens.mass.centre_1 = 0.0
            self.lens_galaxies[0].lens.mass.einstein_radius = 1.6

            self.source_galaxies[0].source.light.centre_0 = 0.0
            self.source_galaxies[0].source.light.centre_1 = 0.0
            self.source_galaxies[0].source.light.intensity = 1.0
            self.source_galaxies[0].source.light.effective_radius = 0.5
            self.source_galaxies[0].source.light.sersic_index = 1.0

    phase1 = SensitivePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                            source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                            sensitive_galaxies=dict(subhalo=gm.GalaxyModel(mass=mp.SphericalIsothermal)),
                            optimizer_class=nl.MultiNest,  phase_name="{}/phase1".format(pipeline_name))

    return pl.PipelineImaging(pipeline_name, phase1)


if __name__ == "__main__":
    pipeline()
