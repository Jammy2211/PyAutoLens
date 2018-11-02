from autolens import conf
from autolens.autofit import non_linear as nl
from autolens.autofit import model_mapper as mm
from autolens.imaging import image as im
from autolens.galaxy import galaxy_model as gm
from autolens.inversion import pixelizations as pix
from autolens.inversion import regularization as reg
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.plotting import imaging_plotters
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp

import os

# Let's go back to our complex source pipeline, but this time, as you've probably guessed, fit it with an inversion.
# As we discussed in the previous tutorial, we'll begin by modeling the source with a light profile, to initialize the
# mass model, and then switch over to an inversion.

# First, lets get our path.
path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

# Lets quickly sort the output directory
conf.instance = conf.Config(config_path=conf.CONFIG_PATH, output_path=path+"output")

def simulate():

    from autolens.imaging import mask
    from autolens.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grids = mask.ImagingGrids.grids_for_simulation(shape=(180, 180), pixel_scale=0.05, psf_shape=(11, 11))

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
                                                 image_plane_grids=[image_plane_grids])

    return im.PreparatoryImage.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.05,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets simulate the image we'll fit, which is the same complex source as the 3_pipelines.py tutorial.
image = simulate()
imaging_plotters.plot_image_subplot(image=image)

def make_pipeline():

    pipeline_name = '6_inversion'

    # This is te same phase 1 as the complex_source pipeline, which we saw will give a good fit to the overall
    # structur of the lensed source and provide a good mass model.

    phase1 = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_1_initialize')

    # Now, in phase 2, lets pass the lens mass model to phase 2, and use it to fit the source with an inversion.
    # We can customize the inversion's priors like we do our light and mass profiles.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            self.lens_galaxies.lens = previous_results[0].variable.lens
            self.source_galaxies.source.pixelization.shape_0 = mm.UniformPrior(lower_limit=20.0, upper_limit=40.0)
            self.source_galaxies.source.pixelization.shape_1 = mm.UniformPrior(lower_limit=20.0, upper_limit=40.0)

            # We know the regularization coefficient is going to be around 1.0 from the last tutorial, but lets use a
            # large prior range just to make sure we cover solutions for any image.
            self.source_galaxies.source.regularization.coefficients_0 = mm.UniformPrior(lower_limit=0.0, upper_limit=100.0)

    phase2 = InversionPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.Rectangular,
                                                                      regularization=reg.Constant)),
                           optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_2_inversion')

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)


pipeline_inversion = make_pipeline()
pipeline_inversion.run(image=image)