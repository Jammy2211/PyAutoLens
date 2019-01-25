from autofit.optimize import non_linear as nl
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline

def make_pipeline(pipeline_path=''):

    pipeline_name = 'pipeline_complex_source'
    pipeline_path = pipeline_path + pipeline_name

    # To begin, we need to initialize the lens's mass model. We should be able to do this by using a simple source
    # model. It won't fit the complicated structure of the source, but it'll give us a reasonable estimate of the
    # einstein radius and the other lens-mass parameters.

    # This should run fine without any prior-passes. In general, a thick, giant ring of source light is something we
    # can be confident MultiNest will fit without much issue, especially when the lens galaxy's light isn't included
    # such that the parameter space is just 12 parameters.

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, phase_name=pipeline_path + '/phase_1_simple_source')

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.5

    # Now lets add another source component, using the previous model as the initialization on the lens / source
    # parameters. We'll vary the parameters of the lens mass model and first source galaxy component during the fit.

    class X2SourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[0].variable.lens
            self.source_galaxies.source.light_0 = previous_results[0].variable.source.light_0

    # You'll notice I've stop writing 'phase_1_results = previous_results[0]' - we know how
    # the previous results are structured now so lets not clutter our code!

    phase2 = X2SourcePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalExponential,
                                                                       light_1=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest, phase_name=pipeline_path + '/phase_2_x2_source')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.5

    # Now lets do the same again, but with 3 source galaxy components.

    class X3SourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[1].variable.lens
            self.source_galaxies.source.light_0 = previous_results[1].variable.source.light_0
            self.source_galaxies.source.light_1 = previous_results[1].variable.source.light_1

    phase3 = X3SourcePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalExponential,
                                                                      light_1=lp.EllipticalSersic,
                                                                      light_2=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest, phase_name=pipeline_path + '/phase_3_x3_source')

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    # And one more for luck!

    class X4SourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[2].variable.lens
            self.source_galaxies.source.light_0 = previous_results[2].variable.source.light_0
            self.source_galaxies.source.light_1 = previous_results[2].variable.source.light_1
            self.source_galaxies.source.light_2 = previous_results[2].variable.source.light_2

    phase4 = X4SourcePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalExponential,
                                                                      light_1=lp.EllipticalSersic,
                                                                      light_2=lp.EllipticalSersic,
                                                                      light_3=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest, phase_name=pipeline_path + '/phase_4_x4_source')

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_path, phase1, phase2, phase3, phase4)