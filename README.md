# PyAutoLens

PyAutoLens makes it simple to model strong gravitational lenses. It is based on the following papers:

https://arxiv.org/abs/1412.7436<br/>
https://arxiv.org/abs/1708.07377

## Slack

We're building a PyAutoLens community on Slack, so you should contact us on our [Slack channel](https://pyautolens.slack.com/) before getting started with PyAutoLens. Here, I can introduce you to the community, give you the latest update on the software and discuss how best to use PyAutoLens for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

## Installation with Docker

We recommend you install AutoLens using Docker. It makes installation easier by containerising the project.

If you don't have Docker then you can install it by following the guide [here](https://docs.docker.com/install/).

Once you have Docker installed you can download the AutoLens Docker project with the command:

```
docker pull autolens/autolens
```

This command can also be used to update the project.

The project can be run using:

```
docker run -it -e LOCAL_USER_ID=`id -u $USER` -h autolens -p 8888:8888 -p 6006:6006 -v $HOME/autolens_workspace:/home/user/workspace autolens/autolens
```

Once the project is running Docker will provide you with a URL. Copy and paste this URL into your browser, making sure you replace '(PyAutoLens or 127.0.0.1)' with '127.0.0.1'. This will bring up a Jupyter notebook including the 'howtolens' directory which is full of tutorials.

Any changes you make inside the Docker workspace will be saved in the autolens_workspace in your home directory.

## Installation with pip

Installation is also available via pip, however there are a number of dependencies that can be installation difficult, see the file [INSTALL.notes](https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes)

AutoLens requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html) and [Numba](https://github.com/numba/numba).

```
$ pip install autolens
```

## Installation on Mac

Install [conda](https://conda.io/miniconda.html).

Create a conda environment:

```
conda create -n autolens python=3.7 anaconda
```

Install multinest:

```
conda install -c conda-forge multinest
```

Tell matplotlib what backend to use:

```
echo "backend : TKAgg" > ~/.matplotlib/matplotlibrc
```

Install autolens:

```
pip install autolens
```

## Workspace

If you install AutoLens with Docker a workspace will be generated for you in the home directory the first time you run the image. This contains configuration, examples and tutorials. After the first time you run docker the workspace will persist any changes you make and won't be updated again.

If you installed AutoLens with pip or want to get access to the latest workspace then you can download it from [here](https://drive.google.com/open?id=1QOwXBy2CFmdngN35tjQ4AsoiEHKWpoHR).

## Features

PyAutoLens's advanced modeling features include:

- **Pipelines** - build automated analysis pipelines to fit complex lens models to large samples of strong lenses.
- **Inversions** - Reconstruct complex source galaxy morphologies on a variety of pixel-grids.
- **Adaption** - (December 2018) - Adapt the lensing analysis to the features of the observed strong lens imaging.
- **Multi-Plane** - (January 2019) Model multi-plane lenses, including systems with multiple lensed source galaxies.

## HowToLens

Detailed tutorials demonstrating how to use PyAutoLens can be found in the 'workspace/howtolens' folder:

- **Introduction** - An introduction to strong gravitational lensing and PyAutolens, familiarizing you with the interface and project structure.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines and tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.

## Support & Discussion

If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our [Slack channel](https://pyautolens.slack.com/).

## Contributing

If you have any suggestions or would like to contribute please get in touch.

## Python Example

With PyAutoLens, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits a lens galaxy's light, mass and a source galaxy.

```python
from autolens.pipeline import phase as ph
from autofit.core import non_linear as nl
from autolens.data.imaging import image as im
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.fitting.plotters import fitting_plotters
import os

# In this example, we'll generate a phase which fits a lens + source strong lens system.

# First, lets setup the path to this script so we can easily load the example data.
path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

# Now, load the image, noise-map and PSF from the 'data' folder.
image = im.load_imaging_from_path(image_path=path + '/data/image.fits',
                                  noise_map_path=path + '/data/noise_maps.fits',
                                  psf_path=path + '/data/psf.fits', pixel_scale=0.1)

# We're going to model our lens galaxy using a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal ellsoid). We load these profiles from the 'light_profile (lp)' and 'mass_profile (mp)'
# modules (check out the source code to see all the profiles that are available).

# To setup our model galaxies, we use the 'galaxy_model' module and GalaxyModel class. 
# A GalaxyModel represents a galaxy where the parameters of its associated profiles are 
# variable and fitted for by the analysis.
lens_galaxy_model = gm.GalaxyModel(light=lp.EllipticalSersic, mass=mp.EllipticalIsothermal)
source_galaxy_model = gm.GalaxyModel(light=lp.EllipticalSersic)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search (in this case, MultiNest).
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy_model], source_galaxies=[source_galaxy_model],
                                optimizer_class=nl.MultiNest, phase_name='phase_example')

# We run the phase on the image, print the results and plot the fit.
results = phase.run(image)
print(results)
fitting_plotters.plot_fitting_subplot(fit=results.fit)

```

## Credits

[James Nightingale](https://github.com/Jammy2211) - Co-lead developer and PyAutoLens guru.

[Richard Hayes](https://github.com/rhayes777) - Co-lead developer and [PyAutoFit](https://github.com/rhayes777/PyAutoFit) guru.

[Ashley Kelly](https://github.com/AshKelly) - Developer of [pyquad](https://github.com/AshKelly/pyquad) for fast deflections computations.

[Nan Li](https://github.com/linan7788626) - Docker integration & support.

[Andrew Robertson](https://github.com/Andrew-Robertson) - Critical curve and caustic calculations.

[Andrea Enia](https://github.com/AndreaEnia) - Voronoi source-plane plotting tools.
