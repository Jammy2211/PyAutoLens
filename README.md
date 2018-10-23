# PyAutoLens

PyAutoLens makes it simple to model strong gravitational lenses. It is based on the following papers:

https://arxiv.org/abs/1412.7436<br/>
https://arxiv.org/abs/1708.07377

## SLACK

We're building a PyAutoLens community on SLACK, so you should contact us on our [SLACK channel](https://pyautolens.slack.com/) before getting started with PyAutoLens. Here, I can introduce you to the community, give you the latest update on the software and discuss how best to use PyAutoLens for your science case.

Unfortunately, SLACK is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

## Installation

AutoLens requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html) and [Numba](https://github.com/numba/numba).

```
$ pip install autolens
```

Known issues with the installation can be found in the file [INSTALL.notes](https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes)

## Installation with Docker

An easy alternative is to install AutoLens using Docker. It makes installation easier by containerising the project.

If you don't have Docker then you can install it by following the guide [here](https://docs.docker.com/install/).

Once you have Docker installed you can download the AutoLens Docker project with the command:

```
docker pull rhayes777/autolens
```

This command can also be used to update the project.

The project can be run using:

```
docker run -it -e LOCAL_USER_ID=`id -u $USER` -h autolens -p 8888:8888 -p 6006:6006 -v $HOME/autolens_workspace:/home/user/workspace rhayes777/autolens
```

Once the project is running Docker will provide you with a URL. Copy and paste this URL into your browser, making sure you replace '(PyAutoLens or 127.0.0.1)' with '127.0.0.1'. This will bring up a Jupyter notebook including the 'howtolens' directory which is full of tutorials.

Any changes you make inside the Docker workspace will be saved in the autolens_workspace in your home directory.

## Python Example

With PyAutoLens, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits a lens galaxy's light, mass and a source galaxy.

```python
from autolens.pipeline import phase as ph
from autolens.autofit import non_linear as nl
from autolens.lensing import galaxy_prior as gp
from autolens.imaging import image as im
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.plotting import fitting_plotters
import os

# In this example, we'll generate a phase which fits a lens + source plane system.

# First, lets setup the path to this script so we can easily load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Now, load the image, noise-map and PSF from the 'data' folder.
image = im.load_imaging_from_path(image_path=path + '/data/image.fits',
                                  noise_map_path=path + '/data/noise_map.fits',
                                  psf_path=path + '/data/psf.fits', pixel_scale=0.1)

# We're going to model our lens galaxy using a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal sphere). We load these profiles from the 'light_profile (lp)' and 'mass_profile (mp)'
# modules (check out the source code to see all the profiles that are available).

# To setup our model galaxies, we use the 'galaxy_model' module and GalaxyModel class. 
# A GalaxyModel represents a galaxy where the parameters of its associated profiles are 
# variable and fitted for by the analysis.
lens_galaxy_model = gp.GalaxyModel(light=lp.AbstractEllipticalSersic, mass=mp.EllipticalIsothermal)
source_galaxy_model = gp.GalaxyModel(light=lp.AbstractEllipticalSersic)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search (in this case, MultiNest).
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy_model], source_galaxies=[source_galaxy_model],
                                optimizer_class=nl.MultiNest, phase_name='phase_example')

# We run the phase on the image, print the results and plot the fit.
results = phase.run(image)
print(results)
fitting_plotters.plot_fitting_subplot(fit=results.fit)

```
## Advanced Lens Modeling

The example above shows the simplest analysis one can perform in PyAutoLens. PyAutoLens's advanced modeling features include:

- **Pipelines** - build automated analysis pipelines to fit complex lens models to large samples of strong lenses.
- **Inversions** - Reconstruct complex source galaxy morphologies on a variety of pixel-grids.
- **Adaption** - (October 2018) - Adapt the lensing analysis to the features of the observed strong lens imaging.
- **Multi-Plane** - (November 2018) Model multi-plane lenses, including systems with multiple lensed source galaxies.

## HowToLens

Detailed tutorials demonstrating how to use PyAutoLens can be found in the 'howtolens' folder:

- **Introduction** - How to use PyAutolens, familiarizing you with the interface and project structure.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines and tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.

## Support & Discussion

If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our [SLACK channel](https://pyautolens.slack.com/).

## Contributing

If you have any suggestions or would like to contribute please get in touch.
