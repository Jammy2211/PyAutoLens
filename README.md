# AutoLens

AutoLens makes it simple to model strong gravitational lenses.

AutoLens is based on these papers:

https://arxiv.org/abs/1412.7436<br/>
https://arxiv.org/abs/1708.07377

## Advantages

- Run advanced modelling pipelines in a single line of code
- Get results quickly thanks to a high degree of optimisation
- Easily write custom pipelines choosing from a large range of mass and light profiles, pixelisations and optimisers
- Create sophisticated models with the minimum of effort

## Installation

```
$ pip install autolens
```

## CLI Example

AutoLens comes with a Command Line Interface.

A list of available pipelines can be printed using:

```bash
$ autolens pipeline
```

For this example we'll run the *profile* pipeline. This pipeline fits the lens and source light with Elliptical Sersic profiles and the lens mass with a SIE profile.</br>

More information on this pipeline can be displayed using:

```bash
$ autolens pipeline profile --info
```

The pipeline can be run on a specified basic.

```bash
$ autolens pipeline profile --image=image/ --pixel-scale=0.05
```

The folder specified by --basic should contain **basic.fits**, **noise.fits** and **psf.fits**.</br>

Results are placed in the *output* folder. This includes output from the optimiser, as well as images showing the models produced throughout the analysis.

## Python Example

AutoLens can be used to create sophisticated analysis pipelines. The below example demonstrates a simple analysis which attempts to fit the system with two light profiles and a mass profile in a single phase.

```python
from autolens.imaging import image as im
from autolens.pipeline import phase
from autolens.analysis import galaxy_prior
from autolens.profiles import light_profiles, mass_profiles
from autolens.autopipe import non_linear

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
source_lens_phase = phase.ProfileSourceLensPhase(lens_galaxy=lens_galaxy, source_galaxy=source_galaxy,
                                                 optimizer_class=non_linear.MultiNest)

# We run the phase on the image and print the results.
results = source_lens_phase.run(image)

# As well as these results there will be images and plots in the 'output' folder.
print(results)
```

Phases can be made to use different optimisers, source galaxies with pixelisations instead of profiles and much more. Phases can also be tied together into pipelines to optimise and accelerate analysis.


## Contributing

If you have any suggestions or would like to contribute please get in touch.
