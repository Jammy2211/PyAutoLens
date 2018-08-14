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

AutoLens requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html) and [Numba](https://github.com/numba/numba).

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

The pipeline can be run on an image in a specified folder.

```bash
$ autolens pipeline profile --image=image/ --pixel-scale=0.05
```

The folder specified by --image should contain **basic.fits**, **noise.fits** and **psf.fits**.</br>

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

## Pipeline Example

Sophisticated pipelines can be written. These pipelines can fit different components of the image individually, use priors and best fit models from previous phases, prevent overfitting and modify the image on the fly.</br>
The *profile* pipeline described here is built into AutoLens and can be run using the CLI.

```python
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.autopipe import non_linear as nl
from autolens.analysis import galaxy_prior as gp
from autolens.imaging import mask as msk
from autolens.imaging import image as im
from autolens.profiles import light_profiles, mass_profiles

# Load an image from the 'basic' folder. It is assumed that this folder contains image.fits, noise.fits and psf.fits.
img = im.load('basic', pixel_scale=0.05)

# In the first phase we attempt to fit the lens light with an EllipticalSersicLightProfile.
phase1 = ph.LensOnlyPhase(lens_galaxy=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersicLightProfile),
                          optimizer_class=nl.MultiNest)


# In the second phase we remove the lens light found in the first phase and try to fit just the source. To do this we
# extend ProfileSourceLensPhase class and override two methods.
class LensSubtractedPhase(ph.ProfileSourceLensPhase):
    # The modify image method provides a way for us to modify the image before a phase starts.
    def modify_image(self, image, previous_results):
        # The image is the original image we are trying to fit. Previous results is a list of results from previous
        # phases. We access the result from the last phase by calling previous_results.last. We take the image of the
        # lens galaxy from the last phase from the image.
        return image - previous_results.last.lens_galaxy_image

    # The pass prior method provides us with a way to set variables and constants in this phase using those from a
    # previous phase.
    def pass_priors(self, previous_results):
        # Here we set the centre of the mass profile of our lens galaxy to a prior provided by the light profile of the
        # lens galaxy in the previous phase. This means that the centre is still variable, but constrained to a set of
        # likely values in parameter space.
        self.lens_galaxy.sie.centre = previous_results.last.variable.lens_galaxy.elliptical_sersic.centre


# A mask function determines which parts of the image should be masked out in analysis. By default the mask is a disc
# with a radius of 3 arc seconds. Here we are only interested in light from the source galaxy so we define an annular
# mask.
def annular_mask_function(image):
    return msk.Mask.annular(image.shape_arc_seconds, pixel_scale=image.pixel_scale, inner_radius=0.4,
                            outer_radius=3.)


# We create the second phase. It's an instance of the phase class we created above with the custom mask function passed
# in. We have a lens galaxy with an SIE mass profile and a source galaxy with an Elliptical Sersic light profile. The
# centre of the mass profile we pass in here will be constrained by the pass_priors function defined above.
phase2 = LensSubtractedPhase(lens_galaxy=gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermal),
                             source_galaxy=gp.GalaxyPrior(
                                 elliptical_sersic=light_profiles.EllipticalSersicLightProfile),
                             optimizer_class=nl.MultiNest,
                             mask_function=annular_mask_function)


# In the third phase we try fitting both lens and source light together. We use priors determined by both the previous
# phases to constrain parameter space search.
class CombinedPhase(ph.ProfileSourceLensPhase):
    # We can take priors from both of the previous results.
    def pass_priors(self, previous_results):
        # The lens galaxy's light profile is constrained using results from the first phase whilst its mass profile
        # depends on results from the second phase.
        self.lens_galaxy = gp.GalaxyPrior(
            elliptical_sersic=previous_results.first.variable.lens_galaxy.elliptical_sersic,
            sie=previous_results.last.variable.lens_galaxy.sie)
        # The source galaxy is constrained using the second phase.
        self.source_galaxy = previous_results.last.variable.source_galaxy


# We define the lens and source galaxies explicitly in pass_priors so there's no need to pass them in when we make the
# phase3 instance
phase3 = CombinedPhase(optimizer_class=nl.MultiNest)

# Phase3h is used to fit hyper galaxies. These objects are used to scale noise and prevent over fitting.
phase3h = ph.SourceLensHyperGalaxyPhase()


# The final phase tries to fit the whole system again.
class CombinedPhase2(ph.ProfileSourceLensPhase):
    # We take priors for the galaxies from phase 3 and set their hyper galaxies from phase 3h.
    def pass_priors(self, previous_results):
        phase_3_results = previous_results[2]
        # Note that a result has a variable attribute with priors...
        self.lens_galaxy = phase_3_results.variable.lens_galaxy
        self.source_galaxy = phase_3_results.variable.source_galaxy
        # ...and a constant attribute with fixed values. The fixed values are the best fit from the phase in question.
        # We fix the 'hyper_galaxy' property of our lens and source galaxies here.
        self.lens_galaxy.hyper_galaxy = previous_results.last.constant.lens_galaxy.hyper_galaxy
        self.source_galaxy.hyper_galaxy = previous_results.last.constant.source_galaxy.hyper_galaxy


# We create phase4. Once again, both lens and source galaxies are defined in pass_priors so there's no need to pass
# anything into the constructor.
phase4 = CombinedPhase2(optimizer_class=nl.MultiNest)

# We put all the phases together in a pipeline and give it a name.
pipeline = pl.Pipeline("profile_pipeline", phase1, phase2, phase3, phase3h, phase4)

# The pipeline is run on an image.
results = pipeline.run(img)

# Let's print the results.
for result in results:
    print(result)
```


## Contributing

If you have any suggestions or would like to contribute please get in touch.
