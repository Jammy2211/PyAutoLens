from autofit import conf
from autofit.core import non_linear
from autolens.data.imaging import image as im
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.lensing.plotters import lensing_fitting_plotters
from autolens.data.imaging.plotters import imaging_plotters
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

# In this example, we're going to take an regular and find a lens model that provides a good fit_normal to it and we're going
# to do this without any knowledge of what the 'correct' lens model is.

# So, what do I mean by a 'lens model'? The lens model is the combination of light profiles and mass profiles we use to
# represent lens model_galaxy, source model_galaxy and therefore create a tracer_without_subhalo. Thus, to begin, we have to choose the
# parametrization of our lens model. We don't need to specify the values of its light and mass profiles (e.g. the
# origin, einstein_radius, etc.) - only the profiles are. In this example, we'll use the following lens model:

# 1) A spherical Isothermal Sphere (SIS) for the lens model_galaxy's mass.
# 2) A spherical exponential light profile for the source model_galaxy's light.

# I'll let you into a secret - this is the same lens model that I used to simulate the regular we'll fit_normal (but I'm not
# going to tell you the actual parameters I used!).

# So, how do we infer these parameters? Well, we could randomly guess a lens model, corresponding to some
# random set of parameters. We could use this lens model to create a tracer_without_subhalo and fit_normal the regular-datas, and quantify how
# good the fit_normal was using its likelihood (we inspected this in previous tutorial). If we kept guessing lens
# models, eventually we'd find one that provides a good fit_normal (i.e. high likelihood) to the datas!

# It may sound nuts, but this is actually the basis of how lens modeling works. However, we do a lot better than
# random guessing. Instead, we track the likelihood of our previous guesses, and guess more models using combinations
# of parameters that gave higher likelihood solutions previously. The idea is that if a set of parameters provided a
# good fit_normal to the datas, another set with similar values probably will too.

# This is called 'non-linear search' and its a fairly common analysis found in science. Over the next few tutorials,
# we're going to really get our heads around the concept of a non-linear search - this intuition will prove crucial
# in being a successful lens modeler.

# We're going to use a non-linear search called 'MultiNest'. I highly recommend it, and find its great for lens
# modeling. However, for now, lets not worry about the details of how MultiNest actually works. Instead, just
# picture that a non-linear search in PyAutoLens operates as follows:

# 1) Randomly guess a lens model and use its light-profiles and mass-profiles to set up a lens model_galaxy, source model_galaxy
#    and a tracer_without_subhalo.

# 2) Pass this tracer_without_subhalo through the fitting module, generating a model regular and comparing this model regular to the
#    observed strong lens imaging datas. This means that we've computed a likelihood.

# 3) Repeat this many times, using the likelihoods of previous fits (typically those with a high likelihood) to
#    guide us to the lens models with the highest liikelihood.

# We'll use the path to howtolens multiple times, so make sure you set it upt correctly!
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# You're going to see a line like this in every tutorial this chapter. I recommend that for now you just ignore it.
# A non-linear search can take a long time to run (minutes, hours, or days), and this isn't ideal if you want to
# go through the tutorials without having to constant stop for long periods!

# This line over-rides the configuration of the non-linear search such that it computes the solution really fast. To
# do this, I've 'cheated' - I've computed the solution myself and then input it into the config. Nevertheless, it means
# things will run fast for you, meaning you won't suffer long delays doin the tutorials.
#
# This will all become clear at the end of the chapter, so for now just bare in mind that we are taking a short-cut
# to get our non-linear search to run fast!
conf.instance = conf.Config(config_path=path+'configs/1_non_linear_search', output_path=path+"output")

# This function simulates the regular we'll fit_normal in this tutorial.
def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lensing import ray_tracing

    psf = im.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

    image_plane_grids = grids.DataGrids.grids_for_simulation(shape=(130, 130), pixel_scale=0.1, psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6))
    source_galaxy = g.Galaxy(light=lp.SphericalExponential(centre=(0.0, 0.0), intensity=0.2, effective_radius=0.2))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grids=[image_plane_grids])

    image_simulated = im.Image.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                                                   exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

    return image_simulated

# and this calls the function, setting us up with an regular to model. Lets plot it
image = simulate()
imaging_plotters.plot_image_subplot(image=image)

# To setup a lens model, we use the 'galaxy_model' (imported as 'gm') module, to create 'GalaxyModel' objects.

# A GalaxyModel behaves analogously to the Galaxy objects we're now used to. However, whereas for a Galaxy we
# manually specified the value of every parameter of its light-profiles and mass-profiles, for a GalaxyModel
# these are inferred by the non-linear search.

# Lets model the lens model_galaxy with an SIS mass profile (which is what it was simulated with).
lens_galaxy_model = gm.GalaxyModel(mass=mp.SphericalIsothermal)

# Lets model the source model_galaxy with a spherical exponential light profile (again, what it was simulated with).
source_galaxy_model = gm.GalaxyModel(light=lp.SphericalExponential)

# To perform the non-linear search, we set up a phase using the 'phase' module (imported as 'ph').

# A phase takes our model_galaxy models and fits their parameters via a non-linear search (in this case, MultiNest). In this
# example, we have a lens-plane and source-plane, so we use a LensSourcePlanePhase.

# (Just like we could give profiles descriptive names, like 'light', 'bulge' and 'disk', we can do the exact same
# thing with galaxies. This is very good practise - as once we start using complex lens models, you could potentially
# have a lot of galaxies - and this is the best way to keep track of them!)

# (also, just ignore the 'dict' - its necessary syntax but not something you
phase = ph.LensSourcePlanePhase(lens_galaxies=dict(lens_galaxy=lens_galaxy_model),
                                source_galaxies=dict(source_galaxy=source_galaxy_model),
                                optimizer_class=non_linear.MultiNest,
                                phase_name='1_non_linear_search')

# To run the phase, we simply pass it the regular datas we want to fit_normal, and the non-linear search begins! As the phase
# runs, a logger will show you the parameters of the best-fit_normal model.
results = phase.run(image)

# Whilst it is running (which should only take a few minutes), you should checkout the 'AutoLens/howtolens/output'
# folder. This is where the results of the phase are written to your hard-disk (in the '1_non_linear_search'
# folder). When its completed, regular and output will also appear in this folder, meaning that you don't need to keep
# running python code to see the results.

# We can print the results to see the best-fit_normal model parameters
# print(results) # NOTE - this isn't working yet, need to sort out.

# The best-fit_normal solution (i.e. the highest likelihood) is stored in the 'results', which we can plot as per usual.
lensing_fitting_plotters.plot_fitting_subplot(fit=results.fit)

# The fit_normal looks good, and we've therefore found a model pretty close to the one we simulated the regular with (you can
# confirm this yourself if you want, by comparing the inferred parameters to those found in the simulations.py file).

# And with that, we're done - you've successfully modeled your first strong lens with PyAutoLens! Before moving
# onto the next tutorial, I want you to think about the following:

# 1) a non-linear search is often said to search a 'non-linear parameter-space' - why is the term parameter-space used?

# 2) Why is this parameter space 'non-linear'?

# 3) Initially, the non-linear search randomly guesses the values of the parameters. However, it shouldn't 'know' what
#    reasonable values for a parameter are. For example, it doesn't know that a reasonable Einstein radius is between
#    0.0" and 4.0"). How does it know what are reasonable values of parameters to guess?
