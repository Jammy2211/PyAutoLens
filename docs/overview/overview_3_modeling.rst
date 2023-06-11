.. _overview_3_modeling:

Lens Modeling
=============

Lens modeling is the process of taking data of a strong lens (e.g. imaging data from the Hubble Space Telescope or
interferometer data from ALMA) and fitting it with a lens model, to determine the light and mass distributions of the
lens and source galaxies that best represent the observed strong lens.

Dataset
-------

In this example, we model Hubble Space Telescope imaging of a real strong lens system, with our goal to
infer the lens and source galaxy light and mass models that fit the data well!

.. code-block:: python

    dataset_name = "simple__no_lens_light"
    dataset_path = path.join("dataset", "slacs", "slacs2303+1422")

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.05,
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

Here is what the dataset looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/fitting/chi_squared_map.png
  :width: 400
  :alt: Alternative text

We next mask the dataset, to remove the exterior regions of the image that do not contain emission from the lens or
source galaxy.

.. code-block:: python

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

Note how when we plot the ``Imaging`` below, the figure now zooms into the masked region.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/fitting/chi_squared_map.png
  :width: 400
  :alt: Alternative text

PyAutoFit
---------

Lens modeling uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import **PyAutoFit** separately to **PyAutoLens**

.. code-block:: python

    import autofit as af


Model Composition
-----------------

We compose the lens model that we fit to the data using `af.Model` objects.

These behave analogously to `Galaxy` objects but their  `LightProfile` and `MassProfile` parameters are not specified,
they are instead determined by a fitting procedure.

We will fit our strong lens data with two galaxies:

- A lens galaxy with a `Sersic` light profile representing a bulge and an
  `Isothermal` mass profile representing its mass.

- A source galaxy with an `Exponential` light profile representing a disk.

The redshifts of the lens (z=0.155) and source(z=0.517) are fixed.

.. code-block:: python

    # Lens:

    bulge = af.Model(al.lp.Sersic)
    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.155,
        bulge=bulge,
        mass=mass
    )

    # Source:

    disk = af.Model(al.lp.Exponential)

    source = af.Model(al.Galaxy, redshift=0.517, disk=disk)

The `info` attribute of each `Model` component shows the model in a readable format.

.. code-block:: python

    print(lens.info)
    print()
    print(source.info)

This gives the following output:

.. code-block:: bash

    galaxies

We combine the lens and source model galaxies above into a `Collection`, which is the final lens model we will fit.

The reason we create separate `Collection`'s for the `galaxies` and `model` is so that the `model` can be extended to
include other components than just galaxies.


.. code-block:: python

    # Overall Lens Model:

    galaxies = af.Collection(lens=lens, source=source)
    model = af.Collection(galaxies=galaxies)

The `info` attribute shows the model in a readable format.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

galaxies
    lens
        redshift                                 0.5
        bulge
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.3
            ell_comps
                ell_comps_0                      GaussianPrior, mean = 0.0, sigma = 0.5
                ell_comps_1                      GaussianPrior, mean = 0.0, sigma = 0.5
            intensity                            LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            effective_radius                     UniformPrior, lower_limit = 0.0, upper_limit = 30.0
        mass
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.1
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.1
            ell_comps
                ell_comps_0                      GaussianPrior, mean = 0.0, sigma = 0.3
                ell_comps_1                      GaussianPrior, mean = 0.0, sigma = 0.3
            einstein_radius                      UniformPrior, lower_limit = 0.0, upper_limit = 8.0
    source
        redshift                                 1.0
        disk
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.3
            ell_comps
                ell_comps_0                      GaussianPrior, mean = 0.0, sigma = 0.5
                ell_comps_1                      GaussianPrior, mean = 0.0, sigma = 0.5
            intensity                            LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            effective_radius                     UniformPrior, lower_limit = 0.0, upper_limit = 30.0

Non-linear Search
-----------------

We now choose the non-linear search, which is the fitting method used to determine the set of light and mass profile
parameters that best-fit our data.

In this example we use ``dynesty`` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm that is
very effective at lens modeling.

PyAutoLens supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.

The ``path_prefix`` and ``name`` determine the output folders the results are written on hard-disk.

.. code-block:: python

    search = af.DynestyStatic(path_prefix="overview", name="modeling")

The non-linear search fits the lens model by guessing many lens models over and over iteratively, using the models which
give a good fit to the data to guide it where to guess subsequent model. An animation of a non-linear search is shown
below, where initial lens models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true
  :width: 600

**Credit: Amy Etherington**

Analysis
--------

We next create an ``AnalysisImaging`` object, which contains the ``log_likelihood_function`` that the non-linear search
calls to fit the lens model to the data.

.. code-block:: python

    analysis = al.AnalysisImaging(dataset=dataset)

Model-Fit
---------

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
dynesty samples, model parameters, visualization) to hard-disk.

If you are running the code on your machine, you should checkout the `autolens_workspace/output` folder, which is where
the results of the search are written to hard-disk on-the-fly. This includes lens model parameter estimates with
errors non-linear samples and the visualization of the best-fit lens model inferred by the search so far.

.. code-block:: python

    result = search.fit(model=model, analysis=analysis)


Results
-------

Whilst navigating the output folder, you may of noted the results were contained in a folder that appears as a random
collection of characters.

This is the model-fit's unique identifier, which is generated based on the model, search and dataset used by the fit.
Fitting an identical model, search and dataset will generate the same identifier, meaning that rerunning the script
will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset, a new
unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.

The fit above returns a `Result` object, which includes lots of information on the lens model.

The `info` attribute shows the result in a readable format.

.. code-block:: python

    print(result.info)

This gives the following output:

.. code-block:: bash

    Bayesian Evidence                              6333.47023932
    Maximum Log Likelihood                         6382.79198627
    Maximum Log Posterior                          1442056.41248673
    
    model                                          Collection (N=18)
        galaxies                                   Collection (N=18)
            lens                                   Galaxy (N=12)
                bulge                              Sersic (N=7)
                mass                               Isothermal (N=5)
            source                                 Galaxy (N=6)
                disk                               Exponential (N=6)
    
    Maximum Log Likelihood Model:
    
    galaxies
        lens
            bulge
                centre
                    centre_0                       0.369
                    centre_1                       -0.169
                ell_comps
                    ell_comps_0             0.766
                    ell_comps_1             0.061
                intensity                          0.000
                effective_radius                   1.161
                sersic_index                       1.597
            mass
                centre
                    centre_0                       -0.002
                    centre_1                       0.004
                ell_comps
                    ell_comps_0             -0.037
                    ell_comps_1             -0.107
                einstein_radius                    1.616
        source
            disk
                centre
                    centre_0                       -0.002
                    centre_1                       0.000
                ell_comps
                    ell_comps_0             0.165
                    ell_comps_1             -0.025
                intensity                          0.252
                effective_radius                   0.127
    
    
    Summary (3.0 sigma limits):
    
    galaxies
        lens
            bulge
                centre
                    centre_0                       0.0236 (-0.7006, 0.7200)
                    centre_1                       0.0218 (-0.6997, 1.0533)
                ell_comps
                    ell_comps_0             -0.0801 (-0.9960, 0.9758)
                    ell_comps_1             0.0775 (-0.9711, 0.9989)
                intensity                          0.0000 (0.0000, 0.0000)
                effective_radius                   11.2907 (0.0573, 29.6304)
                sersic_index                       2.7800 (0.8359, 4.9234)
            mass
                centre
                    centre_0                       -0.0036 (-0.0081, 0.0010)
                    centre_1                       0.0039 (-0.0003, 0.0087)
                ell_comps
                    ell_comps_0             -0.0368 (-0.0398, -0.0338)
                    ell_comps_1             -0.1079 (-0.1116, -0.1037)
                einstein_radius                    1.6160 (1.6129, 1.6195)
        source
            disk
                centre
                    centre_0                       -0.0024 (-0.0055, 0.0013)
                    centre_1                       0.0003 (-0.0033, 0.0037)
                ell_comps
                    ell_comps_0             0.1669 (0.1430, 0.2035)
                    ell_comps_1             -0.0244 (-0.0408, -0.0035)
                intensity                          0.2499 (0.2401, 0.2587)
                effective_radius                   0.1275 (0.1245, 0.1309)
    
    
    Summary (1.0 sigma limits):
    
    galaxies
        lens
            bulge
                centre
                    centre_0                       0.0236 (-0.2004, 0.2672)
                    centre_1                       0.0218 (-0.2204, 0.2282)
                ell_comps
                    ell_comps_0             -0.0801 (-0.4468, 0.2718)
                    ell_comps_1             0.0775 (-0.3457, 0.4478)
                intensity                          0.0000 (0.0000, 0.0000)
                effective_radius                   11.2907 (3.0980, 19.0891)
                sersic_index                       2.7800 (1.5561, 3.9258)
            mass
                centre
                    centre_0                       -0.0036 (-0.0051, -0.0021)
                    centre_1                       0.0039 (0.0026, 0.0057)
                ell_comps
                    ell_comps_0             -0.0368 (-0.0379, -0.0357)
                    ell_comps_1             -0.1079 (-0.1090, -0.1066)
                einstein_radius                    1.6160 (1.6149, 1.6170)
        source
            disk
                centre
                    centre_0                       -0.0024 (-0.0036, -0.0013)
                    centre_1                       0.0003 (-0.0009, 0.0016)
                ell_comps
                    ell_comps_0             0.1669 (0.1567, 0.1777)
                    ell_comps_1             -0.0244 (-0.0304, -0.0180)
                intensity                          0.2499 (0.2470, 0.2532)
                effective_radius                   0.1275 (0.1265, 0.1285)
    
    instances
    
    galaxies
        lens
            redshift                               0.5
        source
            redshift                               1.0

Below, we print the maximum log likelihood model inferred.

.. code-block:: python

    print(result.max_log_likelihood_instance.galaxies.lens)
    print(result.max_log_likelihood_instance.galaxies.source)

The result contains the full posterior information of our non-linear search, including all parameter samples,
log likelihood values and tools to compute the errors on the lens model. **PyAutoLens** includes visualization tools
for plotting this.

.. code-block:: python

    search_plotter = aplt.DynestyPlotter(samples=result.samples)
    search_plotter.cornerplot()

Here is an example of how a PDF estimated for a lens model appears:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/modeling/cornerplot.png
  :width: 600
  :alt: Alternative text

The result also contains the maximum log likelihood `Tracer` and `FitImaging` objects which can easily be plotted.

.. code-block:: python

    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer, grid=dataset.grid
    )
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

Here's what the tracer and model-fit of the model which maximizes the log likelihood looks like, providing good
residuals and low chi-squared values:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/fitting/subplot_fit.png
  :width: 600
  :alt: Alternative text

A full guide of result objects is contained in the `autolens_workspace/*/imaging/results` package.

The result also contains the maximum log likelihood ``Tracer`` and ``FitImaging`` objects and which can easily be
plotted.

.. code-block:: python

    tracer_plotter = aplt.TracerPlotter(tracer=result.max_log_likelihood_tracer, grid=mask.derive_grid.masked)
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

The script ``autolens_workspace/*/results`` contains a full description of all information contained
in a ``Result``.

Model Customization
-------------------

The model can be fully customized, making it simple to parameterize and fit many different lens models
using any combination of light and mass profiles.

.. code-block:: python

    # Lens:

    bulge = af.Model(al.lp.DevVaucouleurs)
    mass = af.Model(al.mp.Isothermal)

    """
    This aligns the light and mass profile centres in the model, reducing the
    number of free parameter fitted for by Dynesty by 2.
    """
    bulge.centre = mass.centre

    """
    This fixes the lens galaxy light profile's effective radius to a value of
    0.8 arc-seconds, removing another free parameter.
    """
    bulge.effective_radius = 0.8

    """
    This forces the mass profile's einstein radius to be above 1.0 arc-seconds.
    """
    mass.add_assertion(lens.mass.einstein_radius > 1.0)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass
    )

The ``info`` attribute shows the customized lens model.

.. code-block:: python

    print(lens.info)

This gives the following output:

.. code-block:: bash

Model Cookbook
--------------

The readthedocs contain a modeling cookbook which provides a concise reference to all the ways to customize a lens
model: https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

Linear Light Profiles
---------------------

**PyAutoLens** supports 'linear light profiles', where the `intensity` parameters of all parametric components are
solved via linear algebra every time the model is fitted using a process called an inversion. This inversion always
computes `intensity` values that give the best fit to the data (e.g. they maximize the likelihood) given the other
parameter values of the light profile.

The `intensity` parameter of each light profile is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in the example below by 3) and removing
the degeneracies that occur between the `intensity` and other light profile
parameters (e.g. `effective_radius`, `sersic_index`).

For complex models, linear light profiles are a powerful way to simplify the parameter space to ensure the best-fit
model is inferred.

A full descriptions of this feature is given in the `linear_light_profiles` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/linear_light_profiles.ipynb

Multi Gaussian Expansion
------------------------

A natural extension of linear light profiles are basis functions, which group many linear light profiles together in
order to capture complex and irregular structures in a galaxy's emission.

Using a clever model parameterization a basis can be composed which corresponds to just N = 4-6 parameters, making
model-fitting efficient and robust.

A full descriptions of this feature is given in the ``multi_gaussian_expansion`` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/multi_gaussian_expansion.ipynb

Shapelets
---------

**PyAutoLens** also supports Shapelets, which are a powerful way to fit the light of the galaxies which
typically act as the source galaxy in strong lensing systems.

A full descriptions of this feature is given in the ``shapelets`` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/shapelets.ipynb

Pixelizations
-------------

The source galaxy can be reconstructed using adaptive pixel-grids (e.g. a Voronoi mesh or Delaunay triangulation),
which unlike light profiles, a multi Gaussian expansion or shapelets are not analytic functions that conform to
certain symmetric profiles.

This means they can reconstruct more complex source morphologies and are better suited to performing detailed analyses
of a lens galaxy's mass.

A full descriptions of this feature is given in the ``pixelization`` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/pixelization.ipynb

The fifth overview example of the readthedocs also give a description of pixelizations:

https://pyautolens.readthedocs.io/en/latest/overview/overview_5_pixelizations.html

Wrap-Up
-------

A more detailed description of lens modeling is provided at the following example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/start_here.ipynb

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a non-linear search is and strategies to fit complex lens model to data in efficient and
robust ways.


