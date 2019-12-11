# PyAutoLens

When two or more galaxies are aligned perfectly down our line-of-sight, the background galaxy appears multiple times. This is called strong gravitational lensing, & **PyAutoLens** makes it simple to model strong gravitational lenses, like this one: 

![alt text](https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/gitimage.png)

**PyAutoLens** is based on the following papers:

[Adaptive Semi-linear Inversion of Strong Gravitational Lens Imaging](https://arxiv.org/abs/1412.7436)

[AutoLens: Automated Modeling of a Strong Lens's Light, Mass & Source](https://arxiv.org/abs/1708.07377)

## Python Example

With **PyAutoLens**, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits the foreground lens galaxy's mass & the background source galaxy's light.

```python
import autofit as af
import autolens as al

import os

# In this example, we'll fit a simple lens galaxy + source galaxy system.
dataset_path = '{}/../data/'.format(os.path.dirname(os.path.realpath(__file__)))

lens_name = 'example_lens'

# Get the relative path to the data in our workspace & load the imaging data.
imaging = al.imaging.from_fits(
    image_path=dataset_path + lens_name + '/image.fits',
    psf_path=dataset_path+lens_name+'/psf.fits',
    noise_map_path=dataset_path+lens_name+'/noise_map.fits',
    pixel_scales=0.1)

# Create a mask for the data, which we setup as a 3.0" circle.
mask = al.mask.circular(shape=imaging.shape, pixel_scales=imaging.pixel_scales, radius=3.0)

# We model our lens galaxy using a mass profile (a singular isothermal ellipsoid) & our source galaxy 
# a light profile (an elliptical Sersic).
lens_mass_profile = al.mp.EllipticalIsothermal
source_light_profile = al.lp.EllipticalSersic

# To setup our model galaxies, we use the GalaxyModel class, which represents a galaxy whose parameters 
# are model & fitted for by PyAutoLens. The galaxies are also assigned redshifts.
lens_galaxy_model = al.GalaxyModel(redshift=0.5, mass=lens_mass_profile)
source_galaxy_model = al.GalaxyModel(redshift=1.0, light=source_light_profile)

# To perform the analysis we set up a phase, which takes our galaxy models & fits their parameters using a non-linear
# search (in this case, MultiNest).
phase = al.PhaseImaging(
    galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
    phase_name='example/phase_example', optimizer_class=af.MultiNest)

# We pass the imaging data and mask to the phase, thereby fitting it with the lens model above & plot the resulting fit.
result = phase.run(data=imaging, mask=mask)
al.plot.fit_imaging.subplot(fit=result.most_likely_fit)
```

## Slack

We're building a **PyAutoLens** community on Slack, so you should contact us on our [Slack channel](https://pyautolens.slack.com/) before getting started. Here, I will give you the latest updates on the software & discuss how best to use **PyAutoLens** for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

## Features

**PyAutoLens's** advanced modeling features include:

- **Galaxies** - Use light & mass profiles to make galaxies & perform lensing calculations.
- **Pipelines** - Write automated analysis pipelines to fit complex lens models to large samples of strong lenses.
- **Extended Sources** - Reconstruct complex source galaxy morphologies on a variety of pixel-aa.
- **Adaption** - Adapt the lensing analysis to the features of the observed strong lens imaging.
- **Multi-Plane** - Perform multi-plane ray-tracing & model multi-plane lens systems.
- **Visualization** - Custom visualization libraries for plotting physical lensing quantities & modeling results.

## HowToLens

Included with **PyAutoLens** is the **HowToLens** lecture series, which provides an introduction to strong gravitational lens modeling with **PyAutoLens**. It can be found in the workspace & consists of 4 chapters:

- **Introduction** - An introduction to strong gravitational lensing & **PyAutolens**.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines & tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.

## Workspace

**PyAutoLens** comes with a workspace, which can be found [here](https://github.com/Jammy2211/autolens_workspace) & which includes:

- **Config** - Configuration files which customize the **PyAutoLens** analysis.
- **Data** - Your data folder, including example data-sets distributed with **PyAutoLens**.
- **HowToLens** - The **HowToLens** lecture series.
- **Output** - Where the **PyAutoLens** analysis & visualization are output.
- **Pipelines** - Example pipelines for modeling strong lenses or to use a template for your own pipeline.
- **Plotting** - Scripts enabling customized figures & images.
- **Runners** - Scripts for running a **PyAutoLens** pipeline.
- **Tools** - Tools for simulating strong lens data, creating masks & using many other **PyAutoLens** features.

If you install **PyAutoLens** with conda or pip, you will need to download the workspace from the [autolens_workspace](https://github.com/Jammy2211/autolens_workspace) repository, which is described in the installation instructions below.

## Depedencies

**PyAutoLens** requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html) & [Numba](https://github.com/numba/numba).

## Installation with conda

We recommend installation using a conda environment as this circumvents a number of compatibility issues when installing **PyMultiNest**.

First, install [conda](https://conda.io/miniconda.html).

Create a conda environment:

```
conda create -n autolens python=3.7 anaconda
```

Activate the conda environment:

```
conda activate autolens
```

Install multinest:

```
conda install -c conda-forge multinest
```

Install autolens (v0.32.0 is the most recent stable build):

```
pip install autolens==0.32.0
```

Clone autolens workspace & set WORKSPACE enviroment model:
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
python3 /path/to/autolens_workspace/runners/simple/runner__lens_sie__source_inversion.py
```

## Installation with pip

Installation is also available via pip (v0.32.0 is the most recent stable build), however there are reported issues with installing **PyMultiNest** that can make installation difficult, see the file [INSTALL.notes](https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes)

```
$ pip install autolens==0.32.0
```

Clone autolens workspace & set WORKSPACE enviroment model:
```
cd /path/where/you/want/autolens_workspace
git clone https://github.com/Jammy2211/autolens_workspace
export WORKSPACE=/path/to/autolens_workspace/
```

Set PYTHONPATH to include the autolens_workspace directory:
```
export PYTHONPATH=/path/to/autolens_workspace
```

You can test everything is working by running the example pipeline runner in the autolens_workspace
```
python3 /path/to/autolens_workspace/runners/simple/runner__lens_sie__source_inversion.py
```

## Support & Discussion

If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our [Slack channel](https://pyautolens.slack.com/).

## Contributing

If you have any suggestions or would like to contribute please get in touch.

## Publications

The following papers use **PyAutoLens**:

[Likelihood-free MCMC with Amortized Approximate Likelihood Ratios](https://arxiv.org/abs/1903.04057)

[Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/1909.07346)

[The molecular-gas properties in the gravitationally lensed merger HATLAS J142935.3-002836](https://arxiv.org/abs/1904.00307)

[Galaxy structure with strong gravitational lensing: decomposing the internal mass distribution of massive elliptical galaxies](https://arxiv.org/abs/1901.07801)

[Novel Substructure & Superfluid Dark Matter](https://arxiv.org/abs/1901.03694)

[CO, H2O, H2O+ line & dust emission in a z = 3.63 strongly lensed starburst merger at sub-kiloparsec scales](https://arxiv.org/abs/1903.00273)

## Credits

### Developers

[James Nightingale](https://github.com/Jammy2211) - Lead developer & PyAutoLens guru.

[Richard Hayes](https://github.com/rhayes777) - Lead developer & [PyAutoFit](https://github.com/rhayes777/PyAutoFit) guru.

[Ashley Kelly](https://github.com/AshKelly) - Developer of [pyquad](https://github.com/AshKelly/pyquad) for fast deflections computations.

[Amy Etherington](https://github.com/amyetherington) - Magnification, Critical Curves and Caustic Calculations.

[Xiaoyue Cao](https://github.com/caoxiaoyue) - Analytic Ellipitcal Power-Law Deflection Angle Calculations.

[Qiuhan He] - NFW Profile Lensing Calculations.

[Nan Li](https://github.com/linan7788626) - Docker integration & support.

### Code Donors

[Andrew Robertson](https://github.com/Andrew-Robertson) - Critical curve & caustic calculations.

Mattia Negrello - Visibility models in the uv-plane via direct Fourier transforms.

[Andrea Enia](https://github.com/AndreaEnia) - Voronoi source-plane plotting tools.

[Aristeidis Amvrosiadis](https://github.com/Sketos) - ALMA imaging data loading.
