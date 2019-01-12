from autofit import conf
from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autolens.data import ccd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.data.plotters import ccd_plotters

import os

# Let's go back to our complex source pipeline, but this time, as you've probably guessed, fit it using an inversion.
# As we discussed in tutorial 6, we'll begin by modeling the source with a light profile, to initialize the
# mass model, and then switch to an inversion.

# To setup the config and output paths without docker, you need to uncomment and run the command below. If you are
# using Docker, you don't need to do anything so leave this uncommented!
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(shape=(180, 180), pixel_scale=0.05,
                                                                       psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal( centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0,
                                                         einstein_radius=1.6))
    source_galaxy_0 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=90.0, intensity=0.2,
                                                         effective_radius=1.0, sersic_index=1.5))
    source_galaxy_1 = g.Galaxy(light=lp.EllipticalSersic(centre=(-0.25, 0.25), axis_ratio=0.7, phi=45.0, intensity=0.1,
                                                         effective_radius=0.2, sersic_index=3.0))
    source_galaxy_2 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.45, -0.35), axis_ratio=0.6, phi=90.0, intensity=0.03,
                                                         effective_radius=0.3, sersic_index=3.5))
    source_galaxy_3 = g.Galaxy(light=lp.EllipticalSersic(centre=(-0.05, -0.0), axis_ratio=0.9, phi=140.0, intensity=0.03,
                                                         effective_radius=0.1, sersic_index=4.0))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy],
                                                 source_galaxies=[source_galaxy_0, source_galaxy_1,
                                                                  source_galaxy_2, source_galaxy_3],
                                                 image_plane_grid_stack=image_plane_grid_stack)

    return ccd.CCDData.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.05,
                               exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets simulate the regular we'll fit, which is the same complex source as the
# 'chapter_3_pipelines/tutorial_3_complex_source.py' tutorial.
ccd_data = simulate()
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

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


pipeline_inversion = make_pipeline(pipeline_name='howtolens_c4_t7_6_inversion')
pipeline_inversion.run(data=ccd_data)

# And with that, we now have a pipeline to model strong lenses using an inversion! Checkout the example pipeline in
# 'workspace/pipelines/examples/inversion.py' for an example of an inversion pipeline that includes the lens light
# component.