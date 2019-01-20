from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline

# In this pipeline, we'll perform a basic analysis which fits a source galaxy using an inversion and a
# lens galaxy where its light is not included and fitted, using two phases:

# Phase 1) Fit the lens galaxy's mass (SIE) and source galaxy's light (Sersic).

# Phase 2) Fit the lens galaxy's mass (SIE) and source galaxy's light using an inversion, where the SIE mass model
#          priors are initialized from phase 1.


def make_pipeline(pipeline_name):

    # This is the same phase 1 as the complex source pipeline, which we saw gave a good fit to the overall
    # structure of the lensed source and provided an accurate lens mass model.

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_1_initialize')

    phase1.optimizer.sampling_efficiency = 0.3
    phase1.optimizer.const_efficiency_mode = True

    # Now, in phase 2, lets use the lens mass model to fit the source with an inversion.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            # We can customize the inversion's priors like we do our light and mass profiles.

            self.lens_galaxies.lens = previous_results[0].variable.lens
            self.source_galaxies.source.pixelization.shape_0 = mm.UniformPrior(lower_limit=20.0, upper_limit=40.0)
            self.source_galaxies.source.pixelization.shape_1 = mm.UniformPrior(lower_limit=20.0, upper_limit=40.0)

            # The expected value of the regularization coefficient depends on the details of the data reduction and
            # source galaxy. A broad log-uniform prior is thus an appropriate way to sample the large range of
            # possible values.
            self.source_galaxies.source.regularization.coefficients_0 = mm.LogUniformPrior(lower_limit=1.0e-6,
                                                                                           upper_limit=10000.0)

    phase2 = InversionPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.Rectangular,
                                                                      regularization=reg.Constant)),
                           optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_2_inversion')

    phase2.optimizer.sampling_efficiency = 0.3
    phase2.optimizer.const_efficiency_mode = True

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)