import os

from autofit import conf
from autofit.core import model_mapper as mm
from autofit.core import non_linear as nl

from autolens.model.galaxy import galaxy, galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline import phase as ph
from test.integration import tools

dirpath = os.path.dirname(os.path.realpath(__file__))
conf.instance = conf.Config("{}/../../workspace/config".format(dirpath),
                            "{}/../../workspace/output/".format(dirpath))

dirpath = os.path.dirname(dirpath)
output_path = '{}/output'.format(dirpath)

pipeline_name = "test"


class TestPhaseModelMapper(object):
    def test_pairing_works(self):
        data_name = '/pair_floats'

        tools.reset_paths(data_name, pipeline_name, output_path)

        sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0, effective_radius=1.3,
                                     sersic_index=3.0)

        lens_galaxy = galaxy.Galaxy(light_profile=sersic)

        tools.simulate_integration_image(data_name=data_name, pixel_scale=0.5, lens_galaxies=[lens_galaxy],
                                         source_galaxies=[], target_signal_to_noise=10.0)
        image = tools.load_image(data_name=data_name, pixel_scale=0.5)

        class MMPhase(ph.LensPlanePhase):

            def pass_priors(self, previous_results):
                self.lens_galaxies[0].sersic.intensity = self.lens_galaxies[0].sersic.axis_ratio

        phase = MMPhase(lens_galaxies=dict(lens=gm.GalaxyModel(sersic=lp.EllipticalSersic)),
                        optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(pipeline_name))

        initial_total_priors = phase.variable.prior_count
        phase.make_analysis(image)

        assert phase.lens_galaxies[0].sersic.intensity == phase.lens_galaxies[0].sersic.axis_ratio
        assert initial_total_priors - 1 == phase.variable.prior_count
        assert len(phase.variable.flat_prior_model_tuples) == 1

        lines = list(
            filter(lambda line: "axis_ratio" in line or "intensity" in line, phase.variable.info.split("\n")))

        assert len(lines) == 2
        assert "lens_axis_ratio                                             UniformPrior, lower_limit = 0.2, " \
               "upper_limit = 1.0" in lines
        assert "lens_intensity                                              UniformPrior, lower_limit = 0.2, " \
               "upper_limit = 1.0" in lines

    def test_constants_work(self):
        name = "const_float"
        data_name = '/const_float'

        tools.reset_paths(data_name, name, output_path)

        sersic = lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=1.0, effective_radius=1.3,
                                     sersic_index=3.0)

        lens_galaxy = galaxy.Galaxy(light_profile=sersic)

        tools.simulate_integration_image(data_name=data_name, pixel_scale=0.5, lens_galaxies=[lens_galaxy],
                                         source_galaxies=[], target_signal_to_noise=10.0)
        image = tools.load_image(data_name=data_name, pixel_scale=0.5)

        class MMPhase(ph.LensPlanePhase):

            def pass_priors(self, previous_results):
                self.lens_galaxies[0].sersic.axis_ratio = 0.2
                self.lens_galaxies[0].sersic.phi = 90.0
                self.lens_galaxies[0].sersic.intensity = 1.0
                self.lens_galaxies[0].sersic.effective_radius = 1.3
                self.lens_galaxies[0].sersic.sersic_index = 3.0

        phase = MMPhase(lens_galaxies=[gm.GalaxyModel(sersic=lp.EllipticalSersic)],
                        optimizer_class=nl.MultiNest, phase_name="{}/phase1".format(name))

        phase.optimizer.n_live_points = 20
        phase.optimizer.sampling_efficiency = 0.8

        phase.make_analysis(image)

        sersic = phase.variable.lens_galaxies[0].sersic

        assert isinstance(sersic, mm.PriorModel)

        assert isinstance(sersic.axis_ratio, mm.Constant)
        assert isinstance(sersic.phi, mm.Constant)
        assert isinstance(sersic.intensity, mm.Constant)
        assert isinstance(sersic.effective_radius, mm.Constant)
        assert isinstance(sersic.sersic_index, mm.Constant)
