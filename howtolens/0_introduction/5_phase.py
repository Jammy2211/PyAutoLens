from autolens.autofit import non_linear
from autolens.autofit import model_mapper
from autolens.pipeline import phase as ph
from autolens.lensing import galaxy_model as gm
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import imaging_plotters
from autolens.plotting import fitting_plotters
import os

# In the previous example, we fitted strong lensing data using a tracer. However, because we simulated the image
# ourselves, we knew which combination of light-profiles and mass-profiles produced a good fit.

# In this example, we'll deal the situation that we face in the real world and assume we have no knowledge of what
# solutions actually provide a good fit!

# To do this, we use 'non-linear optimization' algorithms. This operates as follows:

# 1) Randomly guess a 'lens model', where a lens model corresponds to a combination of light-profiles and mass-profiles
#    which are used to create the lens galaxy(s), source galaxy(s) and tracer.

# 2) Pass this tracer through the fitting module, to generate a model image, and compare this model image to the
#    observed strong lens imaging data (producing the residuals, chi-squareds, etc. we saw in the last tutorial).

# 3) Extract a single number measure of the goodness-of-fit of this tracer's fit to the observed imaging data. We'll
#    use the 'likelihood' that we inspected in the previous tutorial.

# 4) Rinse and repeat, sampling many thousands of different models and keeping those which provide the highest
#    likelihood solutions. If we do this enough, we'll eventually hit a lens model that fits the data well.


### NON-LINEAR OPTIMIZATION ####

# Step 4 above isn't strictly true. Non-linear optimizers are a lot cleverer than just randomly guessing lens models
# until something kind of looks like the data. Instead, the employ a strategy when sampling parameter space which
# locates the highest likelihood solutions a lot faster than randomly guessing.

# In this example, we'll use a non-linear optimizer called MultiNest, which is a 'nested sampling algorithm'.
# In nested sampling, parameter space is first mapped out 'broadly'. The sampler then 'focuses in' on the highest
# likelihood regions that were located.

# To read more about non-linear optimization and MultiNest, checkout the link below:
# ?
# ?

# In the file 'simulations_for_phase', we've simulated some images we'll fit in this example.
# I recommend you don't look at the file yet - lets keep the lens models input parameters unknown.
# (However, we'll fit the images with the same mass-profile and light-profile they were made with)

path = "{}".format(os.path.dirname(os.path.realpath(__file__)))
image = im.load_imaging_from_path(image_path=path + '/data/phase_simple_image.fits',
                                  noise_map_path=path + '/data/phase_simple_noise_map.fits',
                                  psf_path=path + '/data/phase_simple_psf.fits', pixel_scale=0.1)
imaging_plotters.plot_image(image=image)

# Visual inspection of the image reveals that the lens galaxy's light was not simulated, thus our lens model need not
# include a lens light component. Thus, our model just needs to include the lens's mass and source's light.

# To setup a lens model, we use the 'galaxy_model' (import as 'gm') module, to create 'GalaxyModel' objects.

# A GalaxyModel behaves analogously to the Galaxy's we've used in previous examples. However, whereas for a Galaxy we
# manually specified the value of every parameter of its light-profiles and mass-profiles, for a GalaxyModel these are
# fitted for by the non-linear optimizer.

# Lets model the lens galaxy with an SIS mass profile (which is what it was simulated with).
lens_galaxy_model = gm.GalaxyModel(mass=mp.SphericalIsothermal)

# Lets model the source galaxy with a spherical exponential light profile (again, what it was simulated with).
source_galaxy_model = gm.GalaxyModel(light=lp.SphericalExponential)

# To perform the non-linear analysis, we set up a phase using the 'phase' module (imported as 'ph').

# A phase takes our galaxiy models and fits their parameters using a non-linear optimizer (in this case, MultiNest).
# In this example, we have a lens-plane and source-plane, so we use a LensSourcePlanePhase.
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy_model], source_galaxies=[source_galaxy_model],
                                optimizer_class=non_linear.MultiNest, phase_name='phase_simple')

# To run the phase, we simply pass it the image data we want to fit, and the non-linear optimization begins!
# (A nice thing about this is that once we setup 1 phase, we can re-use it to fit different images)

# As the phase runs, a logger will show you the parameters of the best-fit model.

# Whilst it is running (which should only take a few minutes), you should checkout the 'AutoLens/output' folder,
# this is where the results of the phase are being written to your hard-disk! When its finished, images and output
# will also appear.
results = phase.run(image)

# We can print the results to see the best-fit model parameters
print(results) # NOTE - this isn't working yet, need to sort out.

# The best-fit solution (i.e. the highest likelihood) is stored in the 'results', which we can plot.
fitting_plotters.plot_fitting(fit=results.fit)

# If a phase has already begun running at its specified path, the analysis resumes from where it was terminated.
# The folder must be manually deleted to start from scratch. So, if you run this script again, you'll notice the
# results appear immediately!
results_1 = phase.run(image)
print(results_1) # NOTE - this isn't working yet, need to sort out.
fitting_plotters.plot_fitting(fit=results_1.fit)

# Finally, lets look at how we can customize a phase using PyAutoLens. Below, we've applied the following customizations
# to a phase (if the use of 'class' in Python is a bit confusing, just ignore it, you can just copy the template
# below to customize phases):

# 1) We've fixed the centre of the lens galaxy's mass-profile's to the coordinates (0.1", 0.1").
#    For real lens modeling, this might be done using the centre of the lens galaxy's light.

# 2) We've changed our prior on the lens galaxy's mass-profile's einstein radius, to a UniformPrior between 1.4" and
#    1.8". This means our non-linear optimizer only samples values of einstein radius between these values.
#    For real lens modeling, this might be done by visually estimating the size of the lens's arc / ring.

# 3) We now use an annulus mask. Given the lens galaxy's light was no present, theres no point including the central
#    regions of the image in the analysis.
#    For real lens modeling, this might occur when you've subtracted the lens galaxy's light and are only interested in
#    the properties of the lens galaxy's mass or the source galaxy.

class CustomPhase(ph.LensSourcePlanePhase):

    def pass_priors(self, previous_results):

        # The 0 index here in 'lens_galaxies[0]' signifies it is the first lens galaxy in the list we input to create
        # the custom phase below.

        # The word 'mass' corresponds to the word we used when setting up the GalaxyModel above.

        self.lens_galaxies[0].mass.centre_0 = 0.1
        self.lens_galaxies[0].mass.centre_1 = 0.1

        # Overwrite the prior on the einstein radius.
        # The 'model_mapper' module is what we use to link our GalaxyModels to the non-linear optimizer.

        self.lens_galaxies[0].mass.einstein_radius = model_mapper.UniformPrior(lower_limit=1.4, upper_limit=1.8)

        # In this example we haven't customized the source galaxies, but commands like the one below would do so
        # self.source_galaxies[0].light.intensity = 0.5

# The mask function allows us to edit the mask applied to the image in this phase.

def mask_function(img):
    return mask.Mask.annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.4,
                             outer_radius_arcsec=2.8)

# We now create our custom phase. the pass_prior will be called automatically, and we pass this function the
# mask_function above such that the phase uses the annulus mask we created above.

custom_phase = CustomPhase(lens_galaxies=[lens_galaxy_model], source_galaxies=[source_galaxy_model],
                            optimizer_class=non_linear.MultiNest, phase_name='phase_custom')

# Run the phase. You should note that the MultiNest run now only have 5 free parameters, because 2 have been fixed
# (the centre of the mass-profile). By fixing a few parameters, and reducing the prior on the einstein radius, the
# best-fit solution is found a lot faster (i.e. 10 seconds)

results_custom = custom_phase.run(image)
print(results_custom) # NOTE - this isn't working yet, need to sort out.
fitting_plotters.plot_fitting(fit=results_custom.fit)

# Above, we saw that one can manually specify the priors associated with each parameter. This begs the question,
# what priors are used if we don't call the 'pass_priors' function? Checking the directory 'AutoLens/config/priors'.
# In the light-profile.ini and mass-profile.ini config files, the priors assumed for every parameter is specified.
# We've set default values to these config files which typically give the best result, but given that every strong lens
# is different you should experiment with using different priors.

# You've just completed the AutoLens introduction tutorials. Checking the summary on where to go next!