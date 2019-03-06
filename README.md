# PyAutoLens

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy is strongly lensed and appears multiple times or as an Einstein ring of light. **PyAutoLens** makes it simple to model strong gravitational lenses, like this one: 

![alt text](https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/gitimage.png)

**PyAutoLens** is based on the following papers:

[Adaptive Semi-linear Inversion of Strong Gravitational Lens Imaging](https://arxiv.org/abs/1412.7436)

[AutoLens: Automated Modeling of a Strong Lens's Light, Mass and Source](https://arxiv.org/abs/1708.07377)

## Python Example

With **PyAutoLens**, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits a lens galaxy's light, mass and a source galaxy.

```python
from autofit import conf
from autofit.optimize import non_linear as nl
from autolens.pipeline import phase as ph
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.data import ccd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import lens_fit_plotters

import os

# In this example, we'll generate a phase which fits a simple lens + source plane system.

# Get the relative path to the data in our workspace and load the ccd imaging data.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

lens_name = 'example_lens'

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/' + lens_name + '/image.fits', 
                                       psf_path=path+'/data/'+lens_name+'/psf.fits',
                                       noise_map_path=path+'/data/'+lens_name+'/noise_map.fits', 
                                       pixel_scale=0.1)

# Create a mask for the data, which we setup below as a 3.0" circle.
mask = msk.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)

# We model our lens galaxy using a mass profile (a singular isothermal ellipsoid) and our source galaxy 
# a light profile (an elliptical Sersic). We load these profiles from the 'light_profiles (lp)' and 
# 'mass_profiles (mp)' modules.
lens_mass_profile = mp.EllipticalIsothermal
source_light_profile = lp.EllipticalSersic

# To setup our model galaxies, we use the GalaxyModel class, representing a galaxy the parameters of 
# which are variable and fitted for by the analysis.
lens_galaxy_model = gm.GalaxyModel(mass=lens_mass_profile)
source_galaxy_model = gm.GalaxyModel(light=source_light_profile)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search 
# (in this case, MultiNest).
phase = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                optimizer_class=nl.MultiNest, phase_name='example/phase_example')

# We run the phase on the ccd data, print the results and plot the fit.
result = phase.run(data=ccd_data)
lens_fit_plotters.plot_fit_subplot(fit=result.most_likely_fit)

```

## Slack

We're building a **PyAutoLens** community on Slack, so you should contact us on our [Slack channel](https://pyautolens.slack.com/) before getting started. Here, I can introduce you to the community, give you the latest update on the software and discuss how best to use **PyAutoLens** for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

## Features

**PyAutoLens's** advanced modeling features include:

- **Pipelines** - build automated analysis pipelines to fit complex lens models to large samples of strong lenses.
- **Inversions** - Reconstruct complex source galaxy morphologies on a variety of pixel-grids.
- **Adaption** - (February 2019) - Adapt the lensing analysis to the features of the observed strong lens imaging.
- **Multi-Plane** - (April 2019) Model multi-plane lenses, including systems with multiple lensed source galaxies.

## HowToLens

Included with **PyAutoLens** is the **HowToLens** eBook, which provides an introduction to strong gravitational lens modeling with **PyAutoLens**. It can be found in the workspace and consists of 4 chapters:

- **Introduction** - An introduction to strong gravitational lensing and **PyAutolens**.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines and tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.

## Workspace

**PyAutoLens** comes with a workspace, which includes the following:

- **Config** - Configuration files which customize the **PyAutoLens** analysis.
- **Data** - Your data folder, including example data-sets distributed with **PyAutoLens**.
- **HowToLens** - The **HowToLens** eBook.
- **Output** - Where the **PyAutoLens** analysis and visualization are output.
- **Pipelines** - Example pipelines to model a strong lens or use a template for your own pipeline.
- **Plotting** - Scripts enabling customized figures and images.
- **Runners** - Scripts for running a **PyAutoLens** pipeline and analysis.
- **Tools** - Tools for simulating strong lens data, creating masks and using many other **PyAutoLens** features.

If you install **PyAutoLens** with conda or pip the workspace will be included and the latest workspace can be found on the master branch of this git repository.

If you install **PyAutoLens** with Docker a workspace will be generated for you in the home directory the first time you run the image. After the first time you run docker the workspace will persist any changes you make and won't be updated again.

## Depedencies

**PyAutoLens** requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html) and [Numba](https://github.com/numba/numba).

## Installation with conda

We recommend installation using a conda environment as this circumvents a number of compatibility issues when installing **PyMultiNest**.

First, install [conda](https://conda.io/miniconda.html).

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

Set PYTHONPATH to autolens:
```
export PYTHONPATH=/path/to/lib/python3.6/site-packages/autolens/
```

Move workspace and set WORKSPACE enviroment variable:
```
mv /path/to/lib/python3.6/site-packages/autolens/workspace /new/path/to/workspace/
export WORKSPACE=/new/path/to/workspace/
```

## Installation with pip

Installation is also available via pip, however there are reported issues with installing **PyMultiNest** that can make installation difficult, see the file [INSTALL.notes](https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes)

```
$ pip install autolens
```

## Installation with Docker

An alternative to conda and pip is Docker, which makes installation easier by containerising the project.

If you don't have Docker then you can install it by following the guide [here](https://docs.docker.com/install/).

Once you have Docker installed you can download the **PyAutoLens** Docker project with the command:

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

## Support & Discussion

If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our [Slack channel](https://pyautolens.slack.com/).

## Contributing

If you have any suggestions or would like to contribute please get in touch.

## Publications

The following papers use **PyAutoLens**:

[Galaxy structure with strong gravitational lensing: decomposing the internal mass distribution of massive elliptical galaxies](https://arxiv.org/abs/1901.07801)

[Novel Substructure and Superfluid Dark Matter](https://arxiv.org/abs/1901.03694)

## Credits

[James Nightingale](https://github.com/Jammy2211) - Lead developer and PyAutoLens guru.

[Richard Hayes](https://github.com/rhayes777) - Lead developer and [PyAutoFit](https://github.com/rhayes777/PyAutoFit) guru.

[Ashley Kelly](https://github.com/AshKelly) - Developer of [pyquad](https://github.com/AshKelly/pyquad) for fast deflections computations.

[Nan Li](https://github.com/linan7788626) - Docker integration & support.

[Andrew Robertson](https://github.com/Andrew-Robertson) - Critical curve and caustic calculations.

[Andrea Enia](https://github.com/AndreaEnia) - Voronoi source-plane plotting tools.
