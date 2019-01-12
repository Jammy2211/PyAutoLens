from autofit import conf
from autofit.core import non_linear as nl
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy_model as gm
from autolens.lens import lens_data as ld
from autolens.lens import ray_tracing
from autolens.lens import lens_fit
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import lens_fit_plotters

import os

# So far, we've not paid much attention to the source galaxy's morphology. We've assumed its a single-component
# exponential profile, which is a fairly crude assumption. A quick look at any image of a real galaxy reveals a wealth
# of different structures that could be present - bulges, disks, bars, star-forming knots and so on.
# Furthermore, there could be more than one source-galaxy!

# In this example, we'll explore how far we can get trying to_fit a complex source using a pipeline. Fitting complex
# source's is an exercise in diminishing returns. Each component we add to our source model brings with it an extra 5-7,
# parameters. If there are 4 components, or multiple galaxies, we're quickly entering the somewhat nasty regime of 
# 30-40+ parameters in our non-linear search. Even with a pipeline, thats a lot of parameters to try and fit!

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

# Lets simulate the image we'll fit, which is another new image.
ccd_data = simulate()
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# Yep, that's a pretty complex source. There are clearly more than 4 peaks of light - I wouldn't like to guess whether 
# there is one or two sources (or more). You'll also notice I omitted the lens galaxy's light for this system, this is 
# to keep the number of parameters down and the phases running fast, but we wouldn't get such a luxury in a real galaxy.

def make_pipeline(pipeline_name):

    # To begin, we need to initialize the lens's mass model. We should be able to do this by using a simple source
    # model. It won't fit the complicated structure of the source, but it'll give us a reasonable estimate of the
    # einstein radius and the other lens-mass parameters.

    # This should run fine without any prior-passes. In general, a thick, giant ring of source light is something we
    # can be confident MultiNest will fit without much issue, especially when the lens galaxy's light isn't included
    # such that the parameter space is just 12 parameters.

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_1_simple_source')

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
                           optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_2_x2_source')

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
                           optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_3_x3_source')

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
                           optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_4_x4_source')

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)

pipeline_complex_source = make_pipeline(pipeline_name='howtolens_c3_t3_complex_source')
pipeline_complex_source.run(data=ccd_data)

# Okay, so with 4 sources, we still couldn't get a good a fit to the source that didn't leave residuals. The thing is,
# I did infact simulate the lens with 4 sources. This means that there is a 'perfect fit', somewhere in that parameter
# space, that we unfortunately missed using the pipeline above.
#
# Lets confirm this, by manually fitting the ccd data with the true input model

lens_data = ld.LensData(ccd_data=ccd_data, mask=msk.Mask.circular(shape=ccd_data.shape,
                                                                  pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0))

lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0,
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
                                             image_plane_grid_stack=lens_data.grid_stack)

true_fit = lens_fit.fit_lens_data_with_tracer(lens_data=lens_data, tracer=tracer)

lens_fit_plotters.plot_fit_subplot(fit=true_fit)

# And indeed, we see far improved residuals, chi-squareds, etc.

# The morale of this story is that, if the source morphology is complex, there is no way we can build a pipeline to
# fit it. For this tutorial, this was true even though our source model could actually fit the data perfectly. For
# real lenses, the source will be *even more complex* and there is even less hope of getting a good fit :(

# But fear not, PyAutoLens has you covered. In chapter 4, we'll introduce a completely new way to model the source
# galaxy, which addresses the problem faced here. But before that, in the next tutorial we'll discuss how we actually
# pass priors in a pipeline.