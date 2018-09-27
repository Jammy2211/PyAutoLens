# AutoLens

AutoLens makes it simple to model strong gravitational lenses.

AutoLens is based on these papers:

https://arxiv.org/abs/1412.7436<br/>
https://arxiv.org/abs/1708.07377

## Installation

AutoLens requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html) and [Numba](https://github.com/numba/numba).

```
$ pip install numba
$ pip install pymultinest
$ git clone https://github.com/Jammy2211/PyAutoLens
```

## Python Example

AutoLens can model a lens in just a small number of python code. The example below demonstrates a simple analysis which fits the lens galaxy's light, mass and the source galaxy's light.

```python
from autolens.pipeline import phase as ph
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy_prior as gp
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import fitting_plotters
import os

# In this example, we'll generate a phase which fits a lens + source plane system. The example data we fit is
# generated using PyAutoLens, in the 'howtolens/3_simulate.py' tutorial.

# Setup the path of the analysis so we can load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Load an image, its noise-map and PSF from the 'data' folder.
image = im.load_imaging_from_path(image_path=path + '/data/phase_image.fits',
                                  noise_map_path=path + '/data/phase_noise_map.fits',
                                  psf_path=path + '/data/phase_psf.fits', pixel_scale=0.1)

# We're going to model our lens galaxy using a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal sphere). We load these profiles from the 'light_profile (lp)' and 'mass_profile (mp)'
# modules (check out the source code to see all the profiles that are available).

# To setup our model galaxies, we use the 'galaxy_model' module and GalaxyModel class. 
# A GalaxyModel represents a galaxy where the parameters of its associated profiles are 
# variable and fitted for by the analysis.
lens_galaxy_model = gp.GalaxyModel(light=lp.EllipticalSersic, mass=mp.EllipticalIsothermal)
source_galaxy_model = gp.GalaxyModel(light=lp.EllipticalSersic)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear optimizer (in this case, MultiNest).
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy_model], source_galaxies=[source_galaxy_model],
                                optimizer_class=nl.MultiNest, phase_name='phase_example')

# We run the phase on the image, print the results and plot the fit.
results = phase.run(image)
print(results)
fitting_plotters.plot_fitting(fit=results.fit)

```
## Advanced Lens Modeling

- Build pipelines out of phases, enabling automated fitting of complex lens models.
- Reconstruct source galaxies using a variety of pixel-grids.
- Perform multi-plane lens analysis.

# HowToLens

Detailed tutorials demonstrating how to use PyAutoLens can be found in the 'howtolens' folder.

## Contributing

If you have any suggestions or would like to contribute please get in touch.
