"""
Initialize the lens model of a 2-plane lens+source system.

This pipeline is composed of 3 phases:

1) Fit and subtract the lens light using an elliptical Sersic.
2) Fit the source using the lens subtracted image from phase 1, with an SIE mass model (lens) and elliptical
Sersic (source).
3) Fit the lens and source simulataneously, using priors based on the results of phases 1 and 2.
"""

pipeline_name = 'initialize_all'


def make():
    from autolens.pipeline import phase
    from autolens.pipeline import pipeline
    from autofit.core import non_linear as nl
    from autolens.model.galaxy import galaxy_model as gp
    from autolens.model.profiles import light_profiles as lp
    from autolens.model.profiles import mass_profiles as mp

    phase1 = phase.LensSourcePlanePhase(lens_galaxies=[gp.GalaxyModel(light=lp.EllipticalSersic,
                                                                      mass=mp.EllipticalIsothermal)],
                                        source_galaxies=[gp.GalaxyModel(light=lp.EllipticalSersic)],
                                        optimizer_class=nl.MultiNest, phase_name='ph1_fit_all')

    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.6

    return pipeline.PipelineImaging(pipeline_name, phase1)
