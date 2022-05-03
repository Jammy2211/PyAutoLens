.. _overview_3_modeling:

Lens Modeling
=============

We can use a ``Tracer`` to fit data of a strong lens and quantify its goodness-of-fit via a
*log_likelihood*.

Of course, when observe an image of a strong lens, we have no idea what combination of
``LightProfile``'s and ``MassProfiles``'s will produce a model-image that looks like the strong lens we observed:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/image.png
  :width: 400
  :alt: Alternative text

The task of finding these ``LightProfiles``'s and ``MassProfiles``'s is called *lens modeling*.

PyAutoFit
---------

Lens modeling with **PyAutoLens** uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import it separately to **PyAutoLens**

.. code-block:: python

    import autofit as af

Model Composition
-----------------

We compose the lens model that we fit to the data using a ``Model`` object, which behaves analogously to the ``Galaxy``,
``LightProfile`` and ``MassProfile`` used previously, however their parameters are not specified and are instead
determined by a fitting procedure.

.. code-block:: python

    lens_galaxy_model = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=al.lp.EllDevVaucouleurs,
        mass=al.mp.EllIsothermal
    )
    source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, disk=al.lp.EllExponential)

We combine the lens and source model galaxies above into a ``Collection``, which is the model we will fit. Note how
we could easily extend this object to compose highly complex models containing many galaxies.

The reason we create separate ``Collection``'s for the ``galaxies`` and ``model`` is because the `model`
can be extended to include other components than just galaxies.

.. code-block:: python

    galaxies = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
    model = af.Collection(galaxies=galaxies)

In this example, we fit our strong lens data with two galaxies:

    - A lens galaxy with a elliptisl Dev Vaucouleurs ``LightProfile`` representing a bulge and
      elliptical isothermal ``MassProfile`` representing its mass.
    - A source galaxy with an elliptical exponential ``LightProfile`` representing a disk.

The redshifts of the lens (z=0.5) and source(z=1.0) are fixed.

Non-linear Search
-----------------

We now choose the non-linear search, which is the fitting method used to determine the set of ``LightProfile``
and ``MassProfile`` parameters that best-fit our data by minimizing the *residuals* and *chi-squared* values and
maximizing its *log likelihood*.

In this example we use ``dynesty`` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm we find is
very effective at lens modeling.

.. code-block:: python

    search = af.DynestyStatic(name="search_example")

**PyAutoLens** supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.


Analysis
--------

We next create an ``AnalysisImaging`` object, which contains the ``log likelihood function`` that the non-linear
search calls to fit the lens model to the data.

.. code-block:: python

    analysis = al.AnalysisImaging(dataset=imaging)

Model-Fit
---------

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
dynesty samples, model parameters, visualization) to hard-disk.

.. code-block:: python

    result = search.fit(model=model, analysis=analysis)

The non-linear search fits the lens model by guessing many lens models over and over iteratively, using the models which
give a good fit to the data to guide it where to guess subsequent model. An animation of a non-linear search is shown
below, where initial lens models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true
  :width: 600

**Credit: Amy Etherington**

Results
-------

Once a model-fit is running, **PyAutoLens** outputs the results of the search to hard-disk on-the-fly. This includes
lens model parameter estimates with errors non-linear samples and the visualization of the best-fit lens model inferred
by the search so far.

The fit above returns a ``Result`` object, which includes lots of information on the lens model.

Below we print the maximum log likelihood model inferred.

.. code-block:: python

    print(result.max_log_likelihood_instance.galaxies.lens)
    print(result.max_log_likelihood_instance.galaxies.source)

This result contains the full posterior information of our non-linear search, including all
parameter samples, log likelihood values and tools to compute the errors on the lens model.

**PyAutoLens** includes many visualization tools for plotting the results of a non-linear search, for example we can
make a corner plot of the probability density function (PDF):

.. code-block:: python

    dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
    dynesty_plotter.cornerplot()

Here is an example of how a PDF estimated for a lens model appears:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/modeling/cornerplot.png
  :width: 600
  :alt: Alternative text

The result also contains the maximum log likelihood ``Tracer`` and ``FitImaging`` objects and which can easily be
plotted.

.. code-block:: python

    tracer_plotter = aplt.TracerPlotter(tracer=result.max_log_likelihood_tracer, grid=mask.masked_grid)
    tracer_plotter.subplot_tracer()

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_imaging_plotter.subplot_fit_imaging()

Here's what the model-fit of the model which maximizes the log likelihood looks like, providing good residuals and
low chi-squared values:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/subplot_fit.png
  :width: 600
  :alt: Alternative text

The script ``autolens_workspace/notebooks/results`` contains a full description of all information contained
in a ``Result``.

Model Customization
-------------------

The ``Model`` can be fully customized, making it simple to parameterize and fit many different lens models
using any combination of ``LightProfile``'s and ``MassProfile``'s:

.. code-block:: python

    lens_galaxy_model = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=al.lp.EllDevVaucouleurs,
        mass=al.mp.EllIsothermal
    )

    """
    This aligns the light and mass profile centres in the model, reducing the
    number of free parameter fitted for by Dynesty by 2.
    """
    lens_galaxy_model.bulge.centre = lens_galaxy_model.mass.centre

    """
    This fixes the lens galaxy light profile's effective radius to a value of
    0.8 arc-seconds, removing another free parameter.
    """
    lens_galaxy_model.bulge.effective_radius = 0.8

    """
    This forces the mass profile's einstein radius to be above 1.0 arc-seconds.
    """
    lens_galaxy_model.mass.add_assertion(lens_galaxy_model.mass.einstein_radius > 1.0)

The above fit used the non-linear search ``dynesty``, but **PyAutoLens** supports many other methods and their
setting can be easily customized:

Wrap-Up
-------

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a non-linear search is and strategies to fit complex lens model to data in efficient and
robust ways.


