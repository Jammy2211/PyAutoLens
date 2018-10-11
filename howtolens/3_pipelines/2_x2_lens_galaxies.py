from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.autofit import model_mapper as mm
from autolens.autofit import non_linear as nl
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import galaxy_model as gm
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import imaging_plotters

# Up to now, all of the images that we fitted had only one lens galaxy. However we saw in chapter 1 that we can
# create multiple galaxies which each contribute to the strong lensing. Multi-galaxy systems are challenging to
# model, because you're adding an extra 5-10 parameters to the non-linear search and, more problematically, the
# degeneracies between the mass-profiles of two galaxies can be severe.

# However, we can nevertheless break the analysis down using a pipeline and give ourselves a shot at getting a good
# lens model. The approach we're going to take is that we're going to fit as much about each individual lens galaxy
# first, before fitting them simultaneously.

# Up to now, I've put a focus on runners being generalizeable. The pipeline we write in this example is going to be
# the opposite - specific to the image we're modeling. Fitting multiple lens galaxies is really difficult and
# writing a pipeline that we can generalize to many lenses isn't currently possible with PyAutoLens.

# First, lets load and inspect the image. You'll notice that we've upped the pixel_scales to 0.05". The 0.1" we've been
# using up to now isn't high enough resolution to fit a multi-galaxy lensing system very well.
path = '/home/jammy/PyCharm/Projects/AutoLens/howtolens/3_pipelines'
image = im.load_imaging_from_path(image_path=path + '/data/2_x2_lens_galaxies_image.fits',
                                  noise_map_path=path+'/data/2_x2_lens_galaxies_noise_map.fits',
                                  psf_path=path + '/data/2_x2_lens_galaxies_psf.fits', pixel_scale=0.05)

imaging_plotters.plot_image(image=image)

# Okay, so looking at the image, we clearly see two blobs of light, corresponding to our two lens galaxies. We also
# see the source's light is pretty complex - the arcs don't posses the rotational symmetry we're used to seeing
# up to now. Multi-galaxy ray-tracing is just a lot more complicated, which means so is modeling them!

# So, how can we brea the lens modeling up? We can:

# 1) Fit and subtract the light of each lens galaxy individually - this will require some careful masking but is doable.
# 2) Use these results to initialize each lens galaxy's mass-profile.

# So, with this in mind, lets build a pipeline composed as follows:

# 1) Fit the lens galaxy light profile on the left of the image, at coordinates (-1.0, 0.0).
# 2) Fit the lens galaxy light profile on the right of the image, at coordinates (1.0, 0.0).
# 3) Use this lens-subtracted image to fit the source galaxy's light. The mass-profiles of the two lens galaxies can
#    use the results of phases 1 and 2 to initialize its priors.
# 4) Fit all relevent parameters simultaneously, using priors from phases 1, 2 and 3.

# Begin with the make pipeline function
def make_pipeline():

    pipeline_name = 'howtolens/3_pipelines/2_x2_lens_galaxies'  # Give the pipeline a name.

    # This galaxy is at (-1.0, 0.0), so we're going to use a small circular mask centred on its location to fit its
    # light profile. Its important that light from the other lens and source don't contaminate our fit.
    def mask_function(img):
        return mask.Mask.circular(img.shape, pixel_scale=img.pixel_scale, radius_mask_arcsec=0.5, centre=(-1.0, 0.0))

    class LeftLensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            # Lets restrict the prior's on the centres around the pixel we know the galaxy's light centre peaks.
            self.lens_galaxies[0].light.centre_0 = mm.GaussianPrior(mean=-1.0, sigma=0.05)
            self.lens_galaxies[0].light.centre_1 = mm.GaussianPrior(mean= 0.0, sigma=0.05)

            # Given we are only fitting the very central region of the lens galaxy, we don't want to let a parameter 
            # like th Sersic index vary, which changes the light profile structure at large radii. Lets fix it to 4.0.
            self.lens_galaxies[0].light.sersic_index = 4.0

    phase1 = LeftLensPhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                           optimizer_class=nl.MultiNest, mask_function=mask_function,
                           phase_name=pipeline_name+'/phase_1_left_lens_light')

    # Now do the exact same with the lens galaxy on the right at (1.0, 0.0)

    def mask_function(img):
        return mask.Mask.circular(img.shape, pixel_scale=img.pixel_scale, radius_mask_arcsec=0.5, centre=(1.0, 0.0))

    class RightLensPhase(ph.LensPlanePhase):

        def pass_priors(self, previous_results):

            # Note that, there is only 1 galaxy in the phase when we set it up below. This means that in this phase
            # the right-hand lens galaxy is still indexed as 0.

            self.lens_galaxies[0].light.centre_0 = mm.GaussianPrior(mean=1.0, sigma=0.05)
            self.lens_galaxies[0].light.centre_1 = mm.GaussianPrior(mean=0.0, sigma=0.05)
            self.lens_galaxies[0].light.sersic_index = 4.0
            
    phase2 = RightLensPhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                            optimizer_class=nl.MultiNest, mask_function=mask_function,
                            phase_name=pipeline_name+'/phase_2_right_lens_light')


    # In the next phase, we fit the source of the lens subtracted image.
    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            return image - phase_1_results.fit.unmasked_model_profile_image - phase_2_results.fit.unmasked_model_profile_image

        def pass_priors(self, previous_results):

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]

            # We're going to link the centres of the light profiles computed above to the centre of the lens galaxy
            # mass-profiles in this phase. However, we're going to fix the centres of the mass profiles to those
            # centres, to get things running fast.

            # To link two results, such that the parameters are not 'variable' but are instead 'constant', we simply
            # use the 'constant' attribute of the previous results.
            self.lens_galaxies[0].mass.centre_0 = phase_1_results.constant.lens_galaxies.light.centre_0
            self.lens_galaxies[0].mass.centre_1 = phase_1_results.constant.lens_galaxies.light.centre_1

            # (There are now both lens galaxes in the model, so our index runs 0 -> 1.)
            self.lens_galaxies[1].mass.centre_0 = phase_2_results.constant.lens_galaxies.light.centre_0
            self.lens_galaxies[1].mass.centre_1 = phase_2_results.constant.lens_galaxies.light.centre_1

    phase3 = LensSubtractedPhase(lens_galaxies=[gm.GalaxyModel(mass=mp.EllipticalIsothermal),
                                                gm.GalaxyModel(mass=mp.EllipticalIsothermal)],
                                 source_galaxies=[gm.GalaxyModel(light=lp.EllipticalExponential)],
                                 optimizer_class=nl.MultiNest,
                                 phase_name=pipeline_name+'/phase_3_fit_sources')

    class FitAllPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, previous_results):
            
            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            phase_3_results = previous_results[2]

            # Because our results are split over multiple phases, we again need to use a GalaxyModel to set them up.
            self.lens_galaxies[0] = gm.GalaxyModel(light=phase_1_results.variable.lens_galaxies[0].light,
                                                   mass=phase_3_results.variable.lens_galaxies[0].mass)

            # Its also important to keep track of the different lens galaxies indexes. Remember, the index spans values
            # based on what galaxies were in that particular phase. That means that in lens galaxy 0 in phase 2
            # and lens galaxy 1 in phase 3 are the same galaxy.
            self.lens_galaxies[1] = gm.GalaxyModel(light=phase_2_results.variable.lens_galaxies[0].light,
                                                   mass=phase_3_results.variable.lens_galaxies[1].mass)

            # When we pass a a 'variable' galaxy from a previous phase, parameters fixed to constants remain constant.
            # Because centre_0 and centre_1 of the mass profile were fixed to constants in phase 3, they're still
            # constants after the line after. We need to therefore manually over-ride their priors.

            self.lens_galaxies[0].mass.centre_0 = phase_3_results.variable.lens_galaxies[0].mass.centre_0
            self.lens_galaxies[0].mass.centre_1 = phase_3_results.variable.lens_galaxies[0].mass.centre_1
            self.lens_galaxies[1].mass.centre_0 = phase_3_results.variable.lens_galaxies[1].mass.centre_0
            self.lens_galaxies[1].mass.centre_1 = phase_3_results.variable.lens_galaxies[1].mass.centre_1

            #  We also want the Sersc index to a free parameter now, so lets change it from a constant to a variable.
            self.lens_galaxies[0].light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=2.0)
            self.lens_galaxies[1].light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=2.0)

            # Things are much simpler for the source galaxies - just like them togerther!
            self.source_galaxies = phase_3_results.variable.source_galaxies

    phase4 = FitAllPhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic,
                                                       mass=mp.EllipticalIsothermal),
                                        gm.GalaxyModel(light=lp.EllipticalSersic,
                                                       mass=mp.EllipticalIsothermal)],
                                 source_galaxies=[gm.GalaxyModel(light=lp.EllipticalExponential)],
                                 optimizer_class=nl.MultiNest,
                                 phase_name=pipeline_name+'/phase_4_fit_all')

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)


pipeline_x2_galaxies = make_pipeline()
pipeline_x2_galaxies.run(image=image)

# And, we're done. This pipeline takes a while to run, as is the nature of multi-galaxy modeling. Nevertheless, the
# techniques we've learnt above can be applied to systems with even more galaxies, albeit the increases in parameters
# will slow down the non-linear search. Here are some more Q&A's

# 1) This system had two very similar lens galaxies, with comparable amounts of light and mass. How common is this?
#    Does it make it harder to model them?

#    Typically, a 2 galaxy system has 1 massive galaxy (that makes up some 80%-90% of the overall light and mass),
#    accompanied by a smaller satellite. The satellite can't be ignored - it impacts the ray-tracing in a measureable
#    way, but this is a lot less degenerate with the 'main' lens galaxy. This means we can often model the  satellite
#    with much simpler profiles (e.g. spherical profiles). So yes, multi-galaxy systems can often be easier to model.

# 2) It got pretty confusing passing all those priors towards the end of the pipeline there. How do I know I haven't
#    messed up?

#    It does get confusing, I won't lie. The approach I take is, when I'm writing a pipeline, to run it, adding each
#    phase as I add them. In the phase's output directory, you can look at the 'model.info' file to get a list of all
#    the parameters and their priors. There, you can check for sure things are behaviour as you expect. Furthermore,
#    as we learn the common ways for people to write runners in PyAutoLens, we'll introduce some syntactical sugar
#    that makes things easier to follow.