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

This data has had the lens galaxy's light already subtracted from it, in order to make the lens modeling process
faster for this example. Extending the example to include the lens light is simple and shown in other examples.

The image of the data shows certain features which are not present in simulated lens data, for example noise-like
features. These are common preprocessing steps performed on real data, which are described in the `data_preparation`
section of the workspace.

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt

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

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_3_modeling/0_subplot_dataset.png
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

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_3_modeling/1_subplot_dataset.png
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

- A lens galaxy with an `Isothermal` mass profile representing its mass, whose centre is fixed to (0.0", 0.0").

- A source galaxy with an `Exponential` light profile representing a disk.

The redshifts of the lens (z=0.155) and source(z=0.517) are fixed.

.. code-block:: python

    # Lens:

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (0.0, 0.0)

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

    Total Free Parameters = 3

    model                                                 Galaxy (N=3)
        mass                                              Isothermal (N=3)

    redshift                                              0.155
    mass
        centre                                            (0.0, 0.0)
        ell_comps
            ell_comps_0                                   GaussianPrior [3], mean = 0.0, sigma = 0.3
            ell_comps_1                                   GaussianPrior [4], mean = 0.0, sigma = 0.3
        einstein_radius                                   UniformPrior [5], lower_limit = 0.0, upper_limit = 8.0



    Total Free Parameters = 6

    model                                                 Galaxy (N=6)
        disk                                              Exponential (N=6)

    redshift                                              0.517
    disk
        centre
            centre_0                                      GaussianPrior [6], mean = 0.0, sigma = 0.3
            centre_1                                      GaussianPrior [7], mean = 0.0, sigma = 0.3
        ell_comps
            ell_comps_0                                   GaussianPrior [8], mean = 0.0, sigma = 0.3
            ell_comps_1                                   GaussianPrior [9], mean = 0.0, sigma = 0.3
        intensity                                         LogUniformPrior [10], lower_limit = 1e-06, upper_limit = 1000000.0
        effective_radius                                  UniformPrior [11], lower_limit = 0.0, upper_limit = 30.0

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

    Total Free Parameters = 9

    model                                                 Collection (N=9)
        galaxies                                          Collection (N=9)
            lens                                          Galaxy (N=3)
                mass                                      Isothermal (N=3)
            source                                        Galaxy (N=6)
                disk                                      Exponential (N=6)

    galaxies
        lens
            redshift                                      0.155
            mass
                centre                                    (0.0, 0.0)
                ell_comps
                    ell_comps_0                           GaussianPrior [3], mean = 0.0, sigma = 0.3
                    ell_comps_1                           GaussianPrior [4], mean = 0.0, sigma = 0.3
                einstein_radius                           UniformPrior [5], lower_limit = 0.0, upper_limit = 8.0
        source
            redshift                                      0.517
            disk
                centre
                    centre_0                              GaussianPrior [6], mean = 0.0, sigma = 0.3
                    centre_1                              GaussianPrior [7], mean = 0.0, sigma = 0.3
                ell_comps
                    ell_comps_0                           GaussianPrior [8], mean = 0.0, sigma = 0.3
                    ell_comps_1                           GaussianPrior [9], mean = 0.0, sigma = 0.3
                intensity                                 LogUniformPrior [10], lower_limit = 1e-06, upper_limit = 1000000.0
                effective_radius                          UniformPrior [11], lower_limit = 0.0, upper_limit = 30.0

Non-linear Search
-----------------

We now choose the non-linear search, which is the fitting method used to determine the set of light and mass profile
parameters that best-fit our data.

In this example we use ``dynesty`` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm that is
very effective at lens modeling.

PyAutoLens supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.

The ``path_prefix`` and ``name`` determine the output folders the results are written on hard-disk.

We include an input ``number_of_cores``, which when above 1 means that Dynesty uses parallel processing to sample multiple
lens models at once on your CPU.

.. code-block:: python

    search = af.DynestyStatic(path_prefix="overview", name="modeling", number_of_cores=4)

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

Run Times
---------

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single ``instance`` of the lens model to be fitted to
   the dataset such that a log likelihood is returned.

 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.

The log likelihood evaluation time can be estimated before a fit using the ``profile_log_likelihood_function`` method,
which returns two dictionaries containing the run-times and information about the fit.

.. code-block:: python

    run_time_dict, info_dict = analysis.profile_log_likelihood_function(
        instance=model.random_instance()
    )

The overall log likelihood evaluation time is given by the ``fit_time`` key.

For this example, it is ~0.01 seconds, which is extremely fast for lens modeling. More advanced lens
modeling features (e.g. shapelets, multi Gaussian expansions, pixelizations) have slower log likelihood evaluation
times (1-3 seconds), and you should be wary of this when using these features.

The ``run_time_dict`` has a break-down of the run-time of every individual function call in the log likelihood
function, whereas the ``info_dict`` stores information about the data which drives the run-time (e.g. number of
image-pixels in the mask, the shape of the PSF, etc.).

.. code-block:: python

    print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

This gives an output of ~0.01 seconds.

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an
estimate of the number of iterations the non-linear search will perform.

Estimating this quantity is more tricky, as it varies depending on the lens model complexity (e.g. number of parameters)
and the properties of the dataset and model being fitted.

For this example, we conservatively estimate that the non-linear search will perform ~10000 iterations per free
parameter in the model. This is an upper limit, with models typically converging in far fewer iterations.

If you perform the fit over multiple CPUs, you can divide the run time by the number of cores to get an estimate of
the time it will take to fit the model. However, above ~6 cores the speed-up from parallelization is less efficient and
does not scale linearly with the number of cores.

.. code-block:: python

    print(
        "Estimated Run Time Upper Limit (seconds) = ",
        (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
        / search.number_of_cores,
    )

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

Bayesian Evidence                                     -38105.45328689
Maximum Log Likelihood                                -38049.90634989
Maximum Log Posterior                                 757231.20186250

model                                                 Collection (N=9)
    galaxies                                          Collection (N=9)
        lens                                          Galaxy (N=3)
            mass                                      Isothermal (N=3)
        source                                        Galaxy (N=6)
            disk                                      Exponential (N=6)

Maximum Log Likelihood Model:

galaxies
    lens
        mass
            ell_comps
                ell_comps_0                           0.220
                ell_comps_1                           0.067
            einstein_radius                           1.654
    source
        disk
            centre
                centre_0                              -0.295
                centre_1                              0.349
            ell_comps
                ell_comps_0                           -0.028
                ell_comps_1                           -0.299
            intensity                                 0.067
            effective_radius                          0.233


Summary (3.0 sigma limits):

galaxies
    lens
        mass
            ell_comps
                ell_comps_0                           0.2188 (0.2141, 0.2218)
                ell_comps_1                           0.0675 (0.0638, 0.0714)
            einstein_radius                           1.6542 (1.6491, 1.6580)
    source
        disk
            centre
                centre_0                              -0.2946 (-0.2986, -0.2895)
                centre_1                              0.3489 (0.3466, 0.3513)
            ell_comps
                ell_comps_0                           -0.0255 (-0.0424, -0.0080)
                ell_comps_1                           -0.2971 (-0.3126, -0.2810)
            intensity                                 0.0669 (0.0644, 0.0694)
            effective_radius                          0.2334 (0.2289, 0.2394)


Summary (1.0 sigma limits):

galaxies
    lens
        mass
            ell_comps
                ell_comps_0                           0.2188 (0.2174, 0.2202)
                ell_comps_1                           0.0675 (0.0660, 0.0689)
            einstein_radius                           1.6542 (1.6526, 1.6554)
    source
        disk
            centre
                centre_0                              -0.2946 (-0.2960, -0.2929)
                centre_1                              0.3489 (0.3480, 0.3500)
            ell_comps
                ell_comps_0                           -0.0255 (-0.0309, -0.0199)
                ell_comps_1                           -0.2971 (-0.3026, -0.2901)
            intensity                                 0.0669 (0.0662, 0.0677)
            effective_radius                          0.2334 (0.2314, 0.2351)

instances

galaxies
    lens
        redshift                                      0.155
    source
        redshift                                      0.517

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

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_3_modeling/cornerplot.png
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

Here's what the tracer and model-fit of the model which maximizes the log likelihood looks like.

The fit has more significant residuals than the previous tutorial, and it is clear that the lens model cannot fully
capture the complex structure of the lensed source galaxy. Nevertheless, it is sufficient to estimate simple
lens quantities, like the Einstein Mass.

The next examples cover all the features that **PyAutoLens** has to improve the model-fit.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_3_modeling/subplot_tracer.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_3_modeling/subplot_fit.png
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

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/overview_3_modeling/features/linear_light_profiles.ipynb

Multi Gaussian Expansion
------------------------

A natural extension of linear light profiles are basis functions, which group many linear light profiles together in
order to capture complex and irregular structures in a galaxy's emission.

Using a clever model parameterization a basis can be composed which corresponds to just N = 4-6 parameters, making
model-fitting efficient and robust.

A full descriptions of this feature is given in the ``multi_gaussian_expansion`` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/overview_3_modeling/features/multi_gaussian_expansion.ipynb

Shapelets
---------

**PyAutoLens** also supports Shapelets, which are a powerful way to fit the light of the galaxies which
typically act as the source galaxy in strong lensing systems.

A full descriptions of this feature is given in the ``shapelets`` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/overview_3_modeling/features/shapelets.ipynb

Pixelizations
-------------

The source galaxy can be reconstructed using adaptive pixel-grids (e.g. a Voronoi mesh or Delaunay triangulation),
which unlike light profiles, a multi Gaussian expansion or shapelets are not analytic functions that conform to
certain symmetric profiles.

This means they can reconstruct more complex source morphologies and are better suited to performing detailed analyses
of a lens galaxy's mass.

A full descriptions of this feature is given in the ``pixelization`` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/overview_3_modeling/features/pixelization.ipynb

The fifth overview example of the readthedocs also give a description of pixelizations:

https://pyautolens.readthedocs.io/en/latest/overview/overview_5_pixelizations.html

Wrap-Up
-------

A more detailed description of lens modeling is provided at the following example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/overview_3_modeling/start_here.ipynb

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a non-linear search is and strategies to fit complex lens model to data in efficient and
robust ways.


