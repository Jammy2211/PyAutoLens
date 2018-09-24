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
$ pip install numba
$ pip install pymultinest
$ git clone https://github.com/Jammy2211/PyAutoLens
```

## Python Example

AutoLens can be used to create sophisticated analysis pipelines. The below example demonstrates a simple analysis which attempts to fit the lens's light, mass and source's light.

```python
from autolens.pipeline import phase as ph
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy_prior as gp
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import os

# In this example, we'll generate a phase which fits a lens + source plane system. The example data we fit is
# generated using the example in 'simulate/basic.py'.

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an image from the 'phase_basic_data' folder. It is assumed that this folder contains image.fits, noise_map.fits and
# psf.fits - we've included some example data there already.
image = im.load(path=path + '/../data/basic/', pixel_scale=0.07)

# The GalaxyModel class represents a galaxy object, where the parameters of its associated profiles are variable and
# fitted for by the lensing.

# Here, we make a lens galaxy with both a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal sphere). These profiles are loaded from the 'light_profile (lp)' and 'mass_profile (mp)'
# modules, check them out in the source code to see all the profiles you can choose from!
lens_galaxy = gp.GalaxyModel(light=lp.EllipticalSersic, mass=mp.EllipticalIsothermal)

# We make the source galaxy just like the lens galaxy - lets use another Sersic light profile.
source_galaxy = gp.GalaxyModel(light=lp.EllipticalSersic)

# Finally, we need to set up the 'phase' in which the lensing is performed. Depending on the lensing you can choose
# from 3 phases, which represent the number of planes in the lens system (LensPlanePhase, LensSourcePlanePhase,
# MultiPlanePhase). For this examplle, we need a LensSourcePlanePhase.
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                optimizer_class=nl.MultiNest, phase_name='ph_basic')

# We run the phase on the image and print the results.
results = phase.run(image)

# As well as these results there will be images and plots in the 'output' folder.
print(results)

```

Phases can be made to use different optimisers, source galaxies with pixelisations instead of profiles and much more. Phases can also be tied together into pipelines to optimise and accelerate analysis.

## Pipeline Example

Sophisticated pipelines can be written. These pipelines can fit different components of the image individually, use priors and best fit models from previous phases, prevent overfitting and modify the image on the fly.</br>
The *example* pipeline show below breaks the fitting of the lens and source galaxies into separate phases, which ensures a faster and more accurate fit.

```python
from autolens.pipeline import phase
from autolens.pipeline import pipeline
from autolens.autofit import non_linear as nl
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import galaxy_prior as gp
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
import os

pipeline_name = 'pl_basic' # Give the pipeline a name.

def make():

    pipeline.setup_pipeline_path(pipeline_name) # Setup the path using the pipeline name.


    # This line follows the same syntax as our phase did in phase/basic.py. However, as we're breaking the analysis
    # down to only fit the lens's light, that means we use the 'LensPlanePhase' and just specify the lens galaxy.
    phase1 = phase.LensPlanePhase(lens_galaxies=[gp.GalaxyModel(sersic=lp.EllipticalSersic)],
                                  optimizer_class=nl.MultiNest, phase_name='ph1')

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase:
    # 1) We want to fit the lens subtracted image computed in phase 1, instead of the observed image.
    # 2) We want to mask the central regions of this image, where there are residuals from the lens light subtraction.

    # Below, we use the 'modify_image' and 'mask_function' to overwrite the image and mask that are used in this phase.
    # The modify_image function has access to and uses 'previous_results' - these are the results computed in phase 1.
    class LensSubtractedPhase(phase.LensSourcePlanePhase):

        def modify_image(self, masked_image, previous_results):
            phase_1_results = previous_results[0]
            return phase_1_results.lens_subtracted_image

    def mask_function(img):
        return mask.Mask.annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.4,
                                 outer_radius_arcsec=3.)

    # We setup phase 2 just like any other phase, now using the LensSubtracted phase we created above so that our new
    # image and masks are used.
    phase2 = LensSubtractedPhase(lens_galaxies=[gp.GalaxyModel(sie=mp.EllipticalIsothermal)],
                                 source_galaxies=[gp.GalaxyModel(sersic=lp.EllipticalSersic)],
                                 optimizer_class=nl.MultiNest, mask_function=mask_function,
                                 phase_name='ph2')

    # Finally, in phase 3, we want to fit all of the lens and source component simulateously, using the results of
    # phases 1 and 2 to initialize the analysis. To do this, we use the 'pass_priors' function, which allows us to
    # map the previous_results of the lens and source galaxies to the galaxies in this phase.
    # The term 'variable' signifies that when we map these results to setup the GalaxyModel, we want the parameters

    # To still be treated as free parameters that are varied during the fit.

    class LensSourcePhase(phase.LensSourcePlanePhase):

        def pass_priors(self, previous_results):
            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]
            self.lens_galaxies[0] = gp.GalaxyModel(sersic=phase_1_results.variable.lens_galaxies[0].sersic,
                                                   sie=phase_2_results.variable.lens_galaxies[0].sie)
            self.source_galaxies = phase_2_results.variable.source_galaxies

    phase3 = LensSourcePhase(lens_galaxies=[gp.GalaxyModel(sersic=lp.EllipticalSersic,
                                                           sie=mp.EllipticalIsothermal)],
                              source_galaxies=[gp.GalaxyModel(sersic=lp.EllipticalSersic)],
                              optimizer_class=nl.MultiNest, phase_name='ph3')

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)

# Load the image data, make the pipeline above and run it.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))
image = im.load(path=path + '/../data/basic/', pixel_scale=0.07)
pipeline_basic = make()
pipeline_basic.run(image=image)
```


## Contributing

If you have any suggestions or would like to contribute please get in touch.
