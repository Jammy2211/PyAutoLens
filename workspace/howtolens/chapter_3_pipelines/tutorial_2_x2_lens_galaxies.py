from autolens import conf
from autolens.autofit import model_mapper as mm
from autolens.autofit import non_linear as nl
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import imaging_plotters

import os

# Up to now, all of the images that we fitted had only one lens model_galaxy. However we saw in chapter 1 that we can
# create multiple galaxies which each contribute to the strong lensing. Multi-model_galaxy systems are challenging to
# model, because you're adding an extra 5-10 parameters to the non-linear search and, more problematically, the
# degeneracies between the mass-profiles of two galaxies can be severe.

# However, we can nevertheless break the analysis down using a pipeline and give ourselves a shot at getting a good
# lens model. The approach we're going to take is that we're going to fit as much about each individual lens model_galaxy
# first, before fitting them simultaneously.

# Up to now, I've put a focus on runners being generalizeable. The pipeline we write in this example is going to be
# the opposite - specific to the images we're modeling. Fitting multiple lens galaxies is really difficult and
# writing a pipeline that we can generalize to many lenses isn't currently possible with PyAutoLens.

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

    lens_galaxy_0 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, -1.0), axis_ratio=0.8, phi=55.0, intensity=0.1,
                                                       effective_radius=0.8, sersic_index=2.5),
                             mass=mp.EllipticalIsothermal(centre=(1.0, 0.0), axis_ratio=0.7, phi=45.0,
                                                          einstein_radius=1.0))
    lens_galaxy_1 = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 1.0), axis_ratio=0.8, phi=100.0, intensity=0.1,
                                                       effective_radius=0.6, sersic_index=3.0),
                             mass=mp.EllipticalIsothermal(centre=(-1.0, 0.0), axis_ratio=0.8, phi=90.0,
                                                          einstein_radius=0.8))
    source_galaxy = g.Galaxy(light=lp.SphericalExponential(centre=(0.05, 0.15), intensity=0.2, effective_radius=0.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy_0, lens_galaxy_1],
                                                 source_galaxies=[source_galaxy], image_plane_grids=image_plane_grids)

    return im.PreparatoryImage.simulate(array=tracer.image_plane_images_for_simulation, pixel_scale=0.05,
                                        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets simulate the images we'll fit, which is a new images, finally!
image = simulate()

imaging_plotters.plot_image_subplot(image=image)


# Okay, so looking at the images, we clearly see two blobs of light, corresponding to our two lens galaxies. We also
# see the source's light is pretty complex - the arcs don't posses the rotational symmetry we're used to seeing
# up to now. Multi-model_galaxy ray-tracing is just a lot more complicated, which means so is modeling them!

# So, how can we brea the lens modeling up? We can:

# 1) Fit and subtract the light of each lens model_galaxy individually - this will require some careful masking but is doable.
# 2) Use these results to initialize each lens model_galaxy's mass-profile.

# So, with this in mind, lets build a pipeline composed as follows:

# 1) Fit the lens model_galaxy light profile on the left of the images, at coordinates (0.0, -1.0).
# 2) Fit the lens model_galaxy light profile on the right of the images, at coordinates (0.0, 1.0).
# 3) Use this lens-subtracted images to fit the source model_galaxy's light. The mass-profiles of the two lens galaxies can
#    use the results of phases 1 and 2 to initialize its priors.
# 4) Fit all relevent parameters simultaneously, using priors from phases 1, 2 and 3.

# Begin with the make pipeline function
def make_pipeline():
    pipeline_name = '2_x2_lens_galaxies'  # Give the pipeline a name.

    # This model_galaxy is at (-1.0, 0.0), so we're going to use a small circular masks centred on its location to fit its
    # light profile. Its important that light from the other lens and source don't contaminate our fit.
    def mask_function(img):
        return mask.Mask.circular(img.shape, pixel_scale=img.pixel_scale, radius_mask_arcsec=0.5, centre=(0.0, -1.0))

    class LeftLensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            # Lets restrict the prior's on the centres around the pixel we know the model_galaxy's light centre peaks.
            self.lens_galaxies.left_lens.light.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.05)
            self.lens_galaxies.left_lens.light.centre_1 = mm.GaussianPrior(mean=-1.0, sigma=0.05)

            # Given we are only fitting the very central region of the lens model_galaxy, we don't want to let a parameter 
            # like th Sersic index vary, which changes the light profile structure at large radii. Lets fix it to 4.0.
            self.lens_galaxies.left_lens.light.sersic_index = 4.0

    phase1 = LeftLensPhase(lens_galaxies=dict(left_lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest, mask_function=mask_function,
                           phase_name=pipeline_name + '/phase_1_left_lens_light')

    # Now do the exact same with the lens model_galaxy on the right at (0.0, 1.0)

    def mask_function(img):
        return mask.Mask.circular(img.shape, pixel_scale=img.pixel_scale, radius_mask_arcsec=0.5, centre=(0.0, 1.0))

    class RightLensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):
            # Note that, there is only 1 model_galaxy in the phase when we set it up below. This means that in this phase
            # the right-hand lens model_galaxy is still indexed as 0.

            self.lens_galaxies.right_lens.light.centre_0 = mm.GaussianPrior(mean=0.0, sigma=0.05)
            self.lens_galaxies.right_lens.light.centre_1 = mm.GaussianPrior(mean=1.0, sigma=0.05)
            self.lens_galaxies.right_lens.light.sersic_index = 4.0

    phase2 = RightLensPhase(lens_galaxies=dict(right_lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                            optimizer_class=nl.MultiNest, mask_function=mask_function,
                            phase_name=pipeline_name + '/phase_2_right_lens_light')

    # In the next phase, we fit the source of the lens subtracted images.
    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            return image - phase_1_results.fit.unmasked_model_profile_images - \
                   phase_2_results.fit.unmasked_model_profile_images

        def pass_priors(self, previous_results):

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]

            # We're going to link the centres of the light profiles computed above to the centre of the lens model_galaxy
            # mass-profiles in this phase. Because the centres of the mass profiles were fixed in phases 1 and 2,
            # linking them using the 'variable' attribute ensures they stay constant.

            self.lens_galaxies.left_lens.mass.centre_0 = \
                phase_1_results.variable.left_lens.light.centre_0
            self.lens_galaxies.left_lens.mass.centre_1 = \
                phase_1_results.variable.left_lens.light.centre_1

            self.lens_galaxies.right_lens.mass.centre_0 = \
                phase_2_results.variable.right_lens.light.centre_0
            self.lens_galaxies.right_lens.mass.centre_1 = \
                phase_2_results.variable.right_lens.light.centre_1

    phase3 = LensSubtractedPhase(lens_galaxies=dict(left_lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal),
                                                    right_lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                                 optimizer_class=nl.MultiNest,
                                 phase_name=pipeline_name + '/phase_3_fit_sources')

    class FitAllPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            phase_3_results = previous_results[2]

            # Because our results are split over multiple phases, we again need to use a GalaxyModel to set them up.
            self.lens_galaxies.left_lens = gm.GalaxyModel(light=phase_1_results.variable.left_lens.light,
                                                          mass=phase_3_results.variable.left_lens.mass)

            # Its also important to keep track of the different lens galaxies indexes. Remember, the index spans values
            # based on what galaxies were in that particular phase. That means that in lens model_galaxy 0 in phase 2
            # and lens model_galaxy 1 in phase 3 are the same model_galaxy.
            self.lens_galaxies.right_lens = gm.GalaxyModel(light=phase_2_results.variable.right_lens.light,
                                                           mass=phase_3_results.variable.right_lens.mass)

            # When we pass a a 'variable' model_galaxy from a previous phase, parameters fixed to constants remain constant.
            # Because centre_0 and centre_1 of the mass profile were fixed to constants in phase 3, they're still
            # constants after the line after. We need to therefore manually over-ride their priors.

            self.lens_galaxies.left_lens.mass.centre_0 = phase_3_results.variable.left_lens.mass.centre_0
            self.lens_galaxies.left_lens.mass.centre_1 = phase_3_results.variable.left_lens.mass.centre_1
            self.lens_galaxies.right_lens.mass.centre_0 = phase_3_results.variable.right_lens.mass.centre_0
            self.lens_galaxies.right_lens.mass.centre_1 = phase_3_results.variable.right_lens.mass.centre_1

            #  We also want the Sersc index to a free parameter now, so lets change it from a constant to a variable.
            self.lens_galaxies.left_lens.light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=2.0)
            self.lens_galaxies.right_lens.light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=2.0)

            # Things are much simpler for the source galaxies - just like them togerther!
            self.source_galaxies = phase_3_results.variable.source_galaxies

    phase4 = FitAllPhase(lens_galaxies=dict(left_lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                      mass=mp.EllipticalIsothermal),
                                            right_lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                        mass=mp.EllipticalIsothermal)),
                         source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                         optimizer_class=nl.MultiNest,
                         phase_name=pipeline_name + '/phase_4_fit_all')

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)


pipeline_x2_galaxies = make_pipeline()
pipeline_x2_galaxies.run(image=image)

# And, we're done. This pipeline takes a while to run, as is the nature of multi-model_galaxy modeling. Nevertheless, the
# techniques we've learnt above can be applied to systems with even more galaxies, albeit the increases in parameters
# will slow down the non-linear search. Here are some more Q&A's

# 1) This system had two very similar lens galaxies, with comparable amounts of light and mass. How common is this?
#    Does it make it harder to model them?

#    Typically, a 2 model_galaxy system has 1 massive model_galaxy (that makes up some 80%-90% of the overall light and mass),
#    accompanied by a smaller satellite. The satellite can't be ignored - it impacts the ray-tracing in a measureable
#    way, but this is a lot less degenerate with the 'main' lens model_galaxy. This means we can often model the  satellite
#    with much simpler profiles (e.g. spherical profiles). So yes, multi-model_galaxy systems can often be easier to model.

# 2) It got pretty confusing passing all those priors towards the end of the pipeline there, didn't it?

#    It does get confusing, I won't lie. This is why we made galaxies named objects - so that we could call them the
#    'left_lens' and 'right_lens'. It still requires caution when writing the pipeline, but goes to show that if
#    you name your galaxies sensibly you should be able to avoid errors, or spot them quickly when you make them.
