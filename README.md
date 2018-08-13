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

```
$ autolens pipeline
```

For this example we'll run the *profile* pipeline. This pipeline fits the lens and source light with Elliptical Sersic profiles and the lens mass with a SIE profile.</br>

More information on this pipeline can be displayed using:

```
$ autolens pipeline profile --info
```

The pipeline can be run on a specified basic.

```
$ autolens pipeline profile --image=image/ --pixel-scale=0.05
```

The folder specified by --basic should contain **basic.fits**, **noise.fits** and **psf.fits**.</br>

Results are placed in the *output* folder. This includes output from the optimiser, as well as images showing the models produced throughout the analysis.


# PyAutoLens

Welcome to PyAutoLens! A project aiming to make strong gravitational lens modeling openly available to the astronomy community. The project is open-source and comes packaged as a set of easy-to-use Python modules. 

We're still working towards our first stable build, so if you check out of the code make sure you switch to the 'develop' branch to see where we're at ;).

PyAutoLens is based on the two methods below, but we've plans afoot to take it much further than what they achieved:



if You'd like to contribute, feel free to contact me!
