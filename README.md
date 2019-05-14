# PyAutoLens

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy is strongly lensed and appears multiple times or as an Einstein ring of light. **PyAutoLens** makes it simple to model strong gravitational lenses, like this one: 

![alt text](https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/gitimage.png)

**PyAutoLens** is based on the following papers:

[Adaptive Semi-linear Inversion of Strong Gravitational Lens Imaging](https://arxiv.org/abs/1412.7436)

[AutoLens: Automated Modeling of a Strong Lens's Light, Mass and Source](https://arxiv.org/abs/1708.07377)

## Python Example

With **PyAutoLens**, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits a lens galaxy's light, mass and a source galaxy.

```python
from autofit.optimize import non_linear as nl
from autolens.pipeline import phase as ph
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.data import ccd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
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

# Create a mask for the data, which we setup as a 3.0" circle.
mask = msk.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)

# We model our lens galaxy using a mass profile (a singular isothermal ellipsoid) and our source galaxy 
# a light profile (an elliptical Sersic). We load these profiles from the 'light_profiles (lp)' and 
# 'mass_profiles (mp)' modules.
lens_mass_profile = mp.EllipticalIsothermal
source_light_profile = lp.EllipticalSersic

# To setup our model galaxies, we use the GalaxyModel class, which represents a galaxy whose parameters 
# are variable and fitted for by the analysis. The galaxies are also assigned redshifts.
lens_galaxy_model = gm.GalaxyModel(redshift=0.5, mass=lens_mass_profile)
source_galaxy_model = gm.GalaxyModel(redshsift=1.0, light=source_light_profile)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search 
# (in this case, MultiNest).
phase = ph.LensSourcePlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                phase_name='example/phase_example', optimizer_class=nl.MultiNest)

# We run the phase on the ccd data, print the results and plot the fit.
result = phase.run(data=ccd_data)
lens_fit_plotters.plot_fit_subplot(fit=result.most_likely_fit)
```

## Slack

We're building a **PyAutoLens** community on Slack, so you should contact us on our [Slack channel](https://pyautolens.slack.com/) before getting started. Here, I will give you the latest updates on the software and discuss how best to use **PyAutoLens** for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

## Features

**PyAutoLens's** advanced modeling features include:

- **Pipelines** - build automated analysis pipelines to fit complex lens models to large samples of strong lenses.
- **Inversions** - Reconstruct complex source galaxy morphologies on a variety of pixel-grids.
- **Adaption** - (Summer 2019) - Adapt the lensing analysis to the features of the observed strong lens imaging.
- **Multi-Plane** - (Summer 2019) - Model multi-plane lenses, including systems with multiple lensed source galaxies.

## HowToLens

Included with **PyAutoLens** is the **HowToLens** lecture series, which provides an introduction to strong gravitational lens modeling with **PyAutoLens**. It can be found in the workspace and consists of 4 chapters:

- **Introduction** - An introduction to strong gravitational lensing and **PyAutolens**.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines and tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.

## Workspace

**PyAutoLens** comes with a workspace, which can be found [here](https://github.com/Jammy2211/autolens_workspace) and which includes the following:

- **Config** - Configuration files which customize the **PyAutoLens** analysis.
- **Data** - Your data folder, including example data-sets distributed with **PyAutoLens**.
- **HowToLens** - The **HowToLens** lecture series.
- **Output** - Where the **PyAutoLens** analysis and visualization are output.
- **Pipelines** - Example pipelines for modeling strong lenses or to use a template for your own pipeline.
- **Plotting** - Scripts enabling customized figures and images.
- **Runners** - Scripts for running a **PyAutoLens** pipeline.
- **Tools** - Tools for simulating strong lens data, creating masks and using many other **PyAutoLens** features.

If you install **PyAutoLens** with conda or pip, you will need to download the workspace from the [autolens_workspace](https://github.com/Jammy2211/autolens_workspace) repository, which is described in the installation instructions below.

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

Clone autolens workspace and set WORKSPACE enviroment variable:
```
cd /path/where/you/want/autolens_workspace
git clone https://github.com/Jammy2211/autolens_workspace
export WORKSPACE=/path/to/autolens_workspace/
```

Set PYTHONPATH to include the autolens_workspace directory:
```
export PYTHONPATH=/path/to/autolens_workspace/
```

You can test everything is working by running the example pipeline runner in the autolens_workspace
```
python3 /path/to/autolens_workspace/runners/runner.py
```

## Installation with pip

Installation is also available via pip, however there are reported issues with installing **PyMultiNest** that can make installation difficult, see the file [INSTALL.notes](https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes)

```
$ pip install autolens
```

Clone autolens workspace and set WORKSPACE enviroment variable:
```
cd /path/where/you/want/autolens_workspace
git clone https://github.com/Jammy2211/autolens_workspace
export WORKSPACE=/path/to/autolens_workspace/
```

Set PYTHONPATH to include the autolens_workspace directory:
```
export PYTHONPATH=/path/to/autolens_workspace/
```

You can test everything is working by running the example pipeline runner in the autolens_workspace
```
python3 /path/to/autolens_workspace/runners/simple/runner_lens_mass_and_source.py
```

## Support & Discussion

If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our [Slack channel](https://pyautolens.slack.com/).

## Contributing

If you have any suggestions or would like to contribute please get in touch.

## Publications

The following papers use **PyAutoLens**:

[The molecular-gas properties in the gravitationally lensed merger HATLAS J142935.3-002836](https://arxiv.org/abs/1904.00307)

[Galaxy structure with strong gravitational lensing: decomposing the internal mass distribution of massive elliptical galaxies](https://arxiv.org/abs/1901.07801)

[Novel Substructure and Superfluid Dark Matter](https://arxiv.org/abs/1901.03694)

[CO, H2O, H2O+ line and dust emission in a z = 3.63 strongly lensed starburst merger at sub-kiloparsec scales](https://arxiv.org/abs/1903.00273)

## Credits

[James Nightingale](https://github.com/Jammy2211) - Lead developer and PyAutoLens guru.

[Richard Hayes](https://github.com/rhayes777) - Lead developer and [PyAutoFit](https://github.com/rhayes777/PyAutoFit) guru.

[Ashley Kelly](https://github.com/AshKelly) - Developer of [pyquad](https://github.com/AshKelly/pyquad) for fast deflections computations.

[Nan Li](https://github.com/linan7788626) - Docker integration & support.

[Andrew Robertson](https://github.com/Andrew-Robertson) - Critical curve and caustic calculations.

[Andrea Enia](https://github.com/AndreaEnia) - Voronoi source-plane plotting tools.
