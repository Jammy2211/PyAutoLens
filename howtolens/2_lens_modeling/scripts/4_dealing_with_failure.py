from autolens.autofit import non_linear as nl
from autolens.autofit import model_mapper as mm
from autolens.pipeline import phase as ph
from autolens.lensing import galaxy_model as gm
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import fitting_plotters

# We finished the last tutorial on sour note. Our non-linear search failed miserably, and we were unable to infer a
# lens model which fitted our realistic data-set well. In this tutorial, we're going to right our past wrongs and infer
# the correct model - not just once, but three times!

### Approach 1 -  Prior Tuning ###

# The first approach we're going to take is we're going to give our non-linear search a helping hand. Lets think about
# our priors - what they're doing is telling the non-linear search where to look in parameter space. If we tell it to
# look in the right place (that is, *tune* our priors), it'll probaly find the best-fit lens model.

# We've already seen that we can fully customize priors in AutoLens, so lets do it. I've set up a custom phase
# below, and specified a new set of priors that'll give the non-linear search a much better chance at inferring the
# correct model. I've also let you know what we're changing the priors from (as are specified by the
# 'config/priors/default' config files.)

# Load the same image as the previous tutorial.
path = '/home/jammy/PyCharm/Projects/AutoLens/howtolens/2_lens_modeling'
image = im.load_imaging_from_path(image_path=path + '/data/3_realism_and_complexity_image.fits',
                                  noise_map_path=path+'/data/3_realism_and_complexity_noise_map.fits',
                                  psf_path=path + '/data/3_realism_and_complexity_psf.fits', pixel_scale=0.1)

class CustomPriorPhase(ph.LensSourcePlanePhase):

    def pass_priors(self, previous_results):

        # We've imported the 'model_mapper' moodule as 'mm' this time, to make the code more readable.

        # By default, the prior on the x and y coordinates of a light / mass profile is a GaussianPrior with mean
        # 0.0" and sigma "1.0. However, visual inspection of our strong lens image tells us that its clearly around
        # x = 0.0" and y = 0.0", so lets reduce where non-linear search looks for these parameters.

        self.lens_galaxies[0].light.centre_0 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
        self.lens_galaxies[0].light.centre_1 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
        self.lens_galaxies[0].mass.centre_0 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
        self.lens_galaxies[0].mass.centre_1 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)

        # By default, the axis-ratio (ellipticity) of our lens galaxy's light pofile is a UniformPrior between 0.2 and
        # 1.0. However, by looking at the image it looks fairly circular, so lets use a GaussianPrior nearer 1.0.
        self.lens_galaxies[0].light.axis_ratio = mm.GaussianPrior(mean=0.8, sigma=0.15)

        # We'll also assume that the light profile's axis_ratio informs us of the mass-profile's axis_ratio, but
        # because this may not strictly be true (because of dark matter) we'll use a wider prior.
        self.lens_galaxies[0].mass.axis_ratio = mm.GaussianPrior(mean=0.8, sigma=0.25)

        # By default, the orientation of the galaxy's light profile, phi, uses a UniformPrior between 0.0 and
        # 180.0 degrees. However, if you look really close at the image (and maybe adjust the color-map of the plot),
        # you'll be able to notice that it is elliptical and that it is oriented around 45.0 degrees counter-clockwise
        # from the x-axis. Lets update our prior
        self.lens_galaxies[0].light.phi = mm.GaussianPrior(mean=45.0, sigma=15.0)

        # Again, lets kind of assume that light's orientation traces that of mass.
        self.lens_galaxies[0].mass.phi = mm.GaussianPrior(mean=45.0, sigma=30.0)

        # The effective radius of a light profile is its 'half-light' radius, the radius at which 50% of its
        # total luminosity is internal to the circle or ellipse defined within that radius. AutoLens assumes a
        # UniformPrior on this quantity between 0.0" and 4.0", but inspection of the image (again, using a colormap
        # scaling) shows the lens's light doesn't extend anywhere near 4.0", so lets reduce it.
        self.lens_galaxies[0].light.effective_radius = mm.GaussianPrior(mean=0.5, sigma=0.8)

        # Typically, we have some knowledge of what morphology our lens galaxy is. Infact, most strong lenses are
        # massive ellipticals, and anyone who studies galaxy morphology will tell you these galaxies have a Sersic index
        # near 4. So lets change our Sersic index from a UniformPrior between 0.8 and 8.0 to reflect this
        self.lens_galaxies[0].light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=1.0)

        # Finally, the 'ring' that the lensed source forms clearly has a radius of about 1.0". This is its Einstein
        # radius, so lets change the prior from a UniformPrior between 0.0" and 4.0".
        self.lens_galaxies[0].mass.einstein_radius = mm.GaussianPrior(mean=1.0, sigma=0.2)

        # In this exercise, I'm not going to change any priors on the source galaxy. Whilst lens modeling experts can
        # look at a strong lens and often tell you roughly where the source-galaxy is be located (in the source-plane),
        # it is something of art form. Furthermore, the source's morphology can be pretty complex and it can become
        # its very diffcult to come up with a good source prior when this is the case.

# We can now create this custom phase and run it. Our non-linear search will start in a much higher likelihood region
# of parameter space. (again, to save you time, I've precomputed the results of this phase, but feel free to run it
# yourself).
custom_prior_phase = CustomPriorPhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                    mass=mp.EllipticalIsothermal)],
                                      source_galaxies=[gm.GalaxyModel(light=lp.EllipticalExponential)],
                                      optimizer_class=nl.MultiNest,
                                      phase_name='howtolens/2_lens_modeling/4_custom_priors')
custom_prior_result = custom_prior_phase.run(image=image)

# Bam! We get a good model. The right model. A glorious model! We gave our non-linear search a helping hand, and it
# repaid us in spades!

# Check out the PDF in the 'output/howstolens/4_custom_priors/optimizer/chains/pdfs' folder - what degeneracies do you
# notice between parameters?
fitting_plotters.plot_fitting_subplot(fit=custom_prior_result.fit)

# Okay, so we've learnt that by tuning our priors to the lens we're fitting, we can increase our chance of inferring a
# good lens model. Before moving onto the next approach, lets think about the advantages and disadvantages of prior
# tuning:

# Advantage - We found the maximum likelihood solution in parameter space.
# Advantage - The phase took less time to run, because the non-linear search explored less of parameter space.
# Disadvantage - If we specified one prior incorrectly, the non-linear search would have began and therefore ended at
#                an incorrect solution.
# Disadvantage - Our phase was tailored to this specific strong lens. If we want to fit a large sample of lenses, we'd
#                have to write a custom phase for every single one - this would take up a lot of our time!

### Approach 2 -  Reducing Complexity ###

# Our non-linear searched failed because we made the lens model more realistic and therefore more complex. Maybe we
# can make it less complex, whilst still keeping it fairly realistic? Maybe there are some assumptions we can make
# to reduce the number of lens model parameters and therefore dimensionality of non-linear parameter space?

# Well, we're scientists, so we can *always* make assumptions. Below, I'm going to create a phase that assumes that
# light-traces-mass. That is, that our light-profile's centre, axis_ratio and orientation are perfectly
# aligned with its mass. This may, or may not, be a reasonable assumption, but it'll remove 4 parameters from the lens
# model (the mass-profiles x, y, axis_ratio and phi), so its worth trying!

class LightTracesMassPhase(ph.LensSourcePlanePhase):

    def pass_priors(self, previous_results):

        # In the pass priors function, we can 'pair' any two parameters by setting them equal to one another. This
        # removes the parameter on the left-hand side of the pairing from the lens model, such that is always assumes
        # the same value as the parameter on the right-hand side.
        self.lens_galaxies[0].mass_centre_0 = self.lens_galaxies[0].light.centre_0

        # Now, the mass-profile's x coordinate will only use the x coordinate of the light profile. Lets do this with
        # the remaining geometric parameters of the light and mass profiles
        self.lens_galaxies[0].mass_centre_1 = self.lens_galaxies[0].light.centre_1
        self.lens_galaxies[0].mass_axis_ratio = self.lens_galaxies[0].light.axis_ratio
        self.lens_galaxies[0].mass_phi = self.lens_galaxies[0].light.phi

# Again, we create this phase and run it. The non-linear search has a less complex parameter space to seach, and thus
# more chance finding its highest likelihood regions. (again, the results have been precomputed for your convience).
light_traces_mass_phase = LightTracesMassPhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                    mass=mp.EllipticalIsothermal)],
                                      source_galaxies=[gm.GalaxyModel(light=lp.EllipticalExponential)],
                                      optimizer_class=nl.MultiNest, phase_name='howtolens/4_light_traces_mass')

light_traces_mass_phase_result = custom_prior_phase.run(image=image)
fitting_plotters.plot_fitting_subplot(fit=light_traces_mass_phase_result.fit)

# The results look pretty good. Our source galaxy fits the data pretty well, and we've clearly inferred a model that
# looks similar to the one above. However, inspection of the residuals shows that the fit wasn't quite as good as the
# custom-phase above.

# It turns out that when I simulated this image, light didn't perfectly trace mass. The light-profile's axis-ratio was
# 0.9, whereas the mass-profiles was 0.8. The quality of the fit has suffered as a result, and the likelihood we've
# inferred is lower.
#
# Herein lies the pitfalls of making assumptions in science - they may make your model less realistic and your results
# worse! Nevertheless, our lens model is clearly much better than it was in the previous tutorial, so making
# assumptions isn't a bad idea if you're struggling to fit the data t all.

# Again, lets consider the advantages and disadvantages of this approach:

# Advantage - By reducing parameter space's complexity, we inferred the global maximum likelihood.
# Advantage - The phase is not specific to one lens - we could run it on many strong lens images.
# Disadvantage - Our model was less realistic, and our fit suffered as a result.


### Approach 3 -  Look Harder ###

# In approaches 1 and 2, we extended our non-linear search an olive branch and generously helped it find the highest
# likelihood regions of parameter space. In approach 3 ,we're going to be stern with our non-linear search, and tell it
# 'look harder'.

# Basically, every non-linear search algorithm has a set of parameters that govern how thoroughly it searches parameter
# space. The more thoroughly it looks, the more like it is that it'll find the global maximum lens model. However,
# the search will also take longer - and we don't want it to take too long to get some results.

# Lets setup a phase, and overwrite some of the non-linear search's parameters from the defaults it assumes in the
# 'config/non_linear.ini' config file:

custom_non_linear_phase = ph.LensSourcePlanePhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                                mass=mp.EllipticalIsothermal)],
                                      source_galaxies=[gm.GalaxyModel(light=lp.EllipticalExponential)],
                                      optimizer_class=nl.MultiNest, phase_name='howtolens/4_custom_non_linear')

# The 'optimizer' below is MultiNest, the non-linear search we're using.

# When MultiNest searches non-linear parameter space, it places down a set of 'live-points', each of which corresponds
# to a particular lens model (set of parameters) and has an associted likelihood. If it guesses a new lens model with a
# higher likelihood than any of the currently active live points, this new lens model will become an active point. As a
# result, the active point with the lowest likelihood is discarded.

# The more live points MultiNest uses, the more thoroughly it will sample parameter space. Lets increase the number of
# points from the default value (50) to 100.
custom_non_linear_phase.optimizer.n_live_points = 100

# When MultiNest thinks its found a 'peak' likelihood in parameter space, it begins to converge around this peak. It
# does this by guessing lens models with similar parameters. However, this peak might not be the global maximum,
# and if MultiNest converges too quickly around a peak it won't realise this before its too late.
#
# The sampling efficiency therefore describes how quickly MultiNest converges around a peak. It assumes values between
# 0.0 and 1.0, where 1.0 corresponds to the fastest convergance but highest risk of not locating the global maximum.
# Lets reduce the sampling efficiency from 0.8 to 0.5.
custom_non_linear_phase.optimizer.sampling_efficiency = 0.5

# These are the two most important MultiNest parameters controlling how it navigates parameter space, so lets run this
# phase and see if our more detailed inspection of parameter space finds the correct lens model.
custom_non_linear_result = custom_prior_phase.run(image=image)

# Indeed, it does. Thus, we can always brute-force our way to a good lens model, if all else fails.
fitting_plotters.plot_fitting_subplot(fit=custom_non_linear_result.fit)

# Finally, lets list the advantages and disadvantages of this approach:

# Advantage - Its easy to setup, we just increase n_live_points or decrease sampling_efficiency.
# Advantage - It generalizes to any strong lens.
# Advantage - We didn't have to make our model less realistic.
# Disadvantage - Its expensive. Very expensive. The run-time of this phase was over 6 hours. For more complex models
#                we could be talking days or weeks (or, dare I say it, months).


# So, there we have it, we can now fit strong lenses PyAutoLens. And if it fails, we know how to get it to work. I hope
# you're feeling pretty smug. You might even be thinking 'why should I bother with the rest of these tutorials, if I
# can fit strong a lens already'.

# Well, my friend,  I want you to think about the last disadvantage listed above. If modeling a single lens could
# really take as long as a month, are you really willing to spend your valuable time waiting for this? I'm not, which
# is why I developed AutoLens, and in the next tutorial we'll see how we can get the best of both worlds - realistic,
# complex lens model that take mere hours to infer!

# Before doing that though, I want you to go over the advantages and disadvantages listed above again, and think
# whether we could combine these different approaches to get the best of all worlds.