from autolens.imaging import image as im
from autolens.pipeline import phase
from autolens.analysis import galaxy_prior
from autolens.profiles import light_profiles, mass_profiles

# Load an image from the 'basic' folder. It is assumed that this folder contains image.fits, noise.fits and psf.fits.
image = im.load('basic', pixel_scale=0.05)

# The GalaxyPrior class represents a variable galaxy object. Here we make the source galaxy by creating a galaxy prior
# and passing it the EllipticalSersicLightProfile. The optimiser will create instances of this light profile with
# different values for intensity, centre etc. as it runs.
source_galaxy = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersicLightProfile)

# We make a lens galaxy with both mass and light profiles. We call the light profile 'light_profile' and the mass
# profile 'mass_profile' but really they could be called whatever we like.
lens_galaxy = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersicLightProfile,
                                       mass_profile=mass_profiles.SphericalIsothermal)

# A source lens phase performs an analysis on an image using the system we've set up. There are lots of different kinds
# of phase that can be plugged together in sophisticated pipelines but for now we'll run a single phase.
source_lens_phase = phase.ProfileSourceLensPhase(lens_galaxy=lens_galaxy, source_galaxy=source_galaxy)

# We run the phase on the image and print the results.
results = source_lens_phase.run(image)

# As well as these results there will be images and plots in the 'output' folder.
print(results)
