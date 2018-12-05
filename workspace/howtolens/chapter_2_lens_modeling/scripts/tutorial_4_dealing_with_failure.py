from autofit import conf
from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autolens.pipeline import phase as ph
from autolens.model.galaxy import galaxy_model as gm
from autolens.data.imaging import image as im
from autolens.data.imaging.plotters import imaging_plotters
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lensing.plotters import lensing_fitting_plotters

import os


# We finished the last tutorial on sour note. Our non-linear search failed miserably, and we were unable to infer a
# lens model which fitted our realistic datas-set well. In this tutorial, we're going to right our past wrongs and infer
# the correct model - not just once, but three times!

# First, lets get the config / simulation / regular loading out the way - we'll fit_normal the same regular as the previous
# tutorial.

#Setup the path for this run
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'/configs/4_dealing_with_failure', output_path=path+"output")

# Even with my custom config files - the non-linear searches will take a bit of time to run in this tutorial. Whilst you
# are waiting, I would skip ahead to the cells ahead of the phase-run cells, and sit back and think about the comments,
# there's a lot to take in in this tutorial so don't feel that you're in a rush!

# Alternatively, set these running and come back in 10 minutes or so - MultiNest resumes from the existing results on
# your hard-disk, so you can rerun things to get the results instantly!

# Another simulate regular function, for the same regular again.
def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)
    image_plane_grids = grids.DataGrids.grids_for_simulation(shape=(130, 130), pixel_scale=0.1, psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.04,
                                                             effective_radius=0.5, sersic_index=3.5),
                           mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=0.8))

    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.5, phi=90.0, intensity=0.03,
                                                       effective_radius=0.3, sersic_index=3.0))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=[image_plane_grids])

    image_simulated = im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                                   exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

    return image_simulated

# Simulate the regular and set it up.
image = simulate()
imaging_plotters.plot_image_subplot(image=image)

### Approach 1 -  Prior Tuning ###

# The first approach we're going to take is we're going to give our non-linear search a helping hand. Lets think about
# our priors - what they're doing is telling the non-linear search where to look in parameter space. If we tell it to
# look in the right place (that is, *tune* our priors), it'll probaly find the best-fit_normal lens model.

# We've already seen that we can fully customize priors in AutoLens, so lets do it. I've set up a custom phase
# below, and specified a new set of priors that'll give the non-linear search a much better chance at inferring the
# correct model. I've also let you know what we're changing the priors from (as are specified by the
# 'config/priors/default' config files.)

class CustomPriorPhase(ph.LensSourcePlanePhase):

    def pass_priors(self, previous_results):

        # We've imported the 'model_mapper' module as 'mm' this time, to make the code more readable. We've also
        # called our lens model_galaxy 'lens' this time, for shorter more readable code.

        # By default, the prior on the x and y coordinates of a light / mass profile is a GaussianPrior with mean
        # 0.0" and sigma "1.0. However, visual inspection of our strong lens regular tells us that its clearly around
        # x = 0.0" and y = 0.0", so lets reduce where non-linear search looks for these parameters.

        self.lens_galaxies.lens.light.centre_0 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
        self.lens_galaxies.lens.light.centre_1 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
        self.lens_galaxies.lens.mass.centre_0 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
        self.lens_galaxies.lens.mass.centre_1 = mm.UniformPrior(lower_limit=-0.05, upper_limit=0.05)

        # By default, the axis-ratio (ellipticity) of our lens model_galaxy's light pofile is a UniformPrior between 0.2 and
        # 1.0. However, by looking at the regular it looks fairly circular, so lets use a GaussianPrior nearer 1.0.
        self.lens_galaxies.lens.light.axis_ratio = mm.GaussianPrior(mean=0.8, sigma=0.15)

        # We'll also assume that the light profile's axis_ratio informs us of the mass-profile's axis_ratio, but
        # because this may not strictly be true (because of dark matter) we'll use a wider prior.
        self.lens_galaxies.lens.mass.axis_ratio = mm.GaussianPrior(mean=0.8, sigma=0.25)

        # By default, the orientation of the model_galaxy's light profile, phi, uses a UniformPrior between 0.0 and
        # 180.0 degrees. However, if you look really close at the regular (and maybe adjust the color-map of the plot),
        # you'll be able to notice that it is elliptical and that it is oriented around 45.0 degrees counter-clockwise
        # from the x-axis. Lets update our prior
        self.lens_galaxies.lens.light.phi = mm.GaussianPrior(mean=45.0, sigma=15.0)

        # Again, lets kind of assume that light's orientation traces that of mass.
        self.lens_galaxies.lens.mass.phi = mm.GaussianPrior(mean=45.0, sigma=30.0)

        # The effective radius of a light profile is its 'half-light' radius, the radius at which 50% of its
        # total luminosity is internal to the circle or ellipse defined within that radius. AutoLens assumes a
        # UniformPrior on this quantity between 0.0" and 4.0", but inspection of the regular (again, using a colormap
        # scaling) shows the lens's light doesn't extend anywhere near 4.0", so lets reduce it.
        self.lens_galaxies.lens.light.effective_radius = mm.GaussianPrior(mean=0.5, sigma=0.8)

        # Typically, we have some knowledge of what morphology our lens model_galaxy is. Infact, most strong lenses are
        # massive ellipticals, and anyone who studies model_galaxy morphology will tell you these galaxies have a Sersic index
        # near 4. So lets change our Sersic index from a UniformPrior between 0.8 and 8.0 to reflect this
        self.lens_galaxies.lens.light.sersic_index = mm.GaussianPrior(mean=4.0, sigma=1.0)

        # Finally, the 'ring' that the lensed source forms clearly has a radius of about 0.8". This is its Einstein
        # radius, so lets change the prior from a UniformPrior between 0.0" and 4.0".
        self.lens_galaxies.lens.mass.einstein_radius = mm.GaussianPrior(mean=0.8, sigma=0.2)

        # In this exercise, I'm not going to change any priors on the source model_galaxy. Whilst lens modeling experts can
        # look at a strong lens and often tell you roughly where the source-model_galaxy is be located (in the source-plane),
        # it is something of art form. Furthermore, the source's morphology can be pretty complex and it can become
        # its very diffcult to come up with a good source prior when this is the case.

# We can now create this custom phase and run it. Our non-linear search will start in a much higher likelihood region
# of parameter space.
custom_prior_phase = CustomPriorPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                                     mass=mp.EllipticalIsothermal)),
                                      source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                                      optimizer_class=nl.MultiNest,
                                      phase_name='4_custom_priors')
custom_prior_result = custom_prior_phase.run(image=image)

# Bam! We get a good model. The right model. A glorious model! We gave our non-linear search a helping hand, and it
# repaid us in spades!

# Check out the PDF in the 'output/howstolens/4_custom_priors/optimizer/chains/pdfs' folder - what degeneracies do you
# notice between parameters?
lensing_fitting_plotters.plot_fitting_subplot(fit=custom_prior_result.fit)

# Okay, so we've learnt that by tuning our priors to the lens we're fitting, we can increase our chance of inferring a
# good lens model. Before moving onto the next approach, lets think about the advantages and disadvantages of prior
# tuning:

# Advantage - We found the maximum likelihood solution in parameter space.
# Advantage - The phase took less time to run, because the non-linear search explored less of parameter space.
# Disadvantage - If we specified one prior incorrectly, the non-linear search would have began and therefore ended at
#                an incorrect solution.
# Disadvantage - Our phase was tailored to this specific strong lens. If we want to fit_normal a large sample of lenses, we'd
#                have to write a custom phase for every single one - this would take up a lot of our time!

### Approach 2 -  Reducing Complexity ###

# Our non-linear searched failed because we made the lens model more realistic and therefore more complex. Maybe we
# can make it less complex, whilst still keeping it fairly realistic? Maybe there are some assumptions we can make
# to reduce the number of lens model parameters and therefore dimensionality of non-linear parameter space?

# Well, we're scientists, so we can *always* make assumptions. Below, I'm going to create a phase that assumes that
# light-traces-mass. That is, that our light-profile's origin, axis_ratio and orientation are perfectly
# aligned with its mass. This may, or may not, be a reasonable assumption, but it'll remove 4 parameters from the lens
# model (the mass-profiles x, y, axis_ratio and phi), so its worth trying!

class LightTracesMassPhase(ph.LensSourcePlanePhase):

    def pass_priors(self, previous_results):

        # In the pass priors function, we can 'pair' any two parameters by setting them equal to one another. This
        # removes the parameter on the left-hand side of the pairing from the lens model, such that is always assumes
        # the same value as the parameter on the right-hand side.
        self.lens_galaxies.lens.mass.centre_0 = self.lens_galaxies.lens.light.centre_0

        # Now, the mass-profile's x coordinate will only use the x coordinate of the light profile. Lets do this with
        # the remaining geometric parameters of the light and mass profiles
        self.lens_galaxies.lens.mass.centre_1 = self.lens_galaxies.lens.light.centre_1
        self.lens_galaxies.lens.mass.axis_ratio = self.lens_galaxies.lens.light.axis_ratio
        self.lens_galaxies.lens.mass.phi = self.lens_galaxies.lens.light.phi

# Again, we create this phase and run it. The non-linear search has a less complex parameter space to seach, and thus
# more chance finding its highest likelihood regions. (again, the results have been precomputed for your convience).
light_traces_mass_phase = LightTracesMassPhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                                       mass=mp.EllipticalIsothermal)),
                                      source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                                      optimizer_class=nl.MultiNest,
                                               phase_name='4_light_traces_mass')

light_traces_mass_phase_result = light_traces_mass_phase.run(image=image)
lensing_fitting_plotters.plot_fitting_subplot(fit=light_traces_mass_phase_result.fit)

# The results look pretty good. Our source model_galaxy fits the datas pretty well, and we've clearly inferred a model that
# looks similar to the one above. However, inspection of the residuals shows that the fit_normal wasn't quite as good as the
# custom-phase above.

# It turns out that when I simulated this regular, light didn't perfectly trace mass. The light-profile's axis-ratio was
# 0.9, whereas the mass-profiles was 0.8. The quality of the fit_normal has suffered as a result, and the likelihood we've
# inferred is lower.
#
# Herein lies the pitfalls of making assumptions in science - they may make your model less realistic and your results
# worse! Nevertheless, our lens model is clearly much better than it was in the previous tutorial, so making
# assumptions isn't a bad idea if you're struggling to fit_normal the datas t all.

# Again, lets consider the advantages and disadvantages of this approach:

# Advantage - By reducing parameter space's complexity, we inferred the global maximum likelihood.
# Advantage - The phase is not specific to one lens - we could run it on many strong lens regular.
# Disadvantage - Our model was less realistic, and our fit_normal suffered as a result.


### Approach 3 -  Look Harder ###

# In approaches 1 and 2, we extended our non-linear search an olive branch and generously helped it find the highest
# likelihood regions of parameter space. In approach 3 ,we're going to be stern with our non-linear search, and tell it
# 'look harder'.

# Basically, every non-linear search algorithm has a set of parameters that govern how thoroughly it searches parameter
# space. The more thoroughly it looks, the more like it is that it'll find the global maximum lens model. However,
# the search will also take longer - and we don't want it to take too long to get some results.

# Lets setup a phase, and overwrite some of the non-linear search's parameters from the defaults it assumes in the
# 'config/non_linear.ini' config file:

custom_non_linear_phase = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                                mass=mp.EllipticalIsothermal)),
                                      source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalExponential)),
                                      optimizer_class=nl.MultiNest,
                                                  phase_name='4_custom_non_linear')

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
custom_non_linear_result = custom_non_linear_phase.run(image=image)

# Indeed, it does. Thus, we can always brute-force our way to a good lens model, if all else fails.
lensing_fitting_plotters.plot_fitting_subplot(fit=custom_non_linear_result.fit)

# Finally, lets list the advantages and disadvantages of this approach:

# Advantage - Its easy to setup, we just increase n_live_points or decrease sampling_efficiency.
# Advantage - It generalizes to any strong lens.
# Advantage - We didn't have to make our model less realistic.
# Disadvantage - Its expensive. Very expensive. The run-time of this phase was over 6 hours. For more complex models
#                we could be talking days or weeks (or, dare I say it, months).

# So, there we have it, we can now fit_normal strong lenses PyAutoLens. And if it fails, we know how to get it to work. I hope
# you're feeling pretty smug. You might even be thinking 'why should I bother with the rest of these tutorials, if I
# can fit_normal strong a lens already'.

# Well, my friend,  I want you to think about the last disadvantage listed above. If modeling a single lens could
# really take as long as a month, are you really willing to spend your valuable time waiting for this? I'm not, which
# is why I developed AutoLens, and in the next tutorial we'll see how we can get the best of both worlds - realistic,
# complex lens model that take mere hours to infer!

# Before doing that though, I want you to go over the advantages and disadvantages listed above again, and think
# whether we could combine these different approaches to get the best of all worlds.