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


In this example, we therefore fit our strong lens data with two galaxies:

    - A lens galaxy with a elliptisl Dev Vaucouleurs ``LightProfile`` representing a bulge and
      elliptical isothermal ``MassProfile`` representing its mass.
    - A source galaxy with an elliptical exponential ``LightProfile`` representing a disk.

The redshifts of the lens (z=0.5) and source(z=1.0) are fixed.

Printing the ``info`` attribute of the model shows us this is the model we are fitting, and shows us the free parameters and
their priors:

.. code-block:: python

    galaxies = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
    model = af.Collection(galaxies=galaxies)

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
            elliptical_comps
                elliptical_comps_0               GaussianPrior, mean = 0.0, sigma = 0.5
                elliptical_comps_1               GaussianPrior, mean = 0.0, sigma = 0.5
            intensity                            LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            effective_radius                     UniformPrior, lower_limit = 0.0, upper_limit = 30.0
        mass
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.1
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.1
            elliptical_comps
                elliptical_comps_0               GaussianPrior, mean = 0.0, sigma = 0.3
                elliptical_comps_1               GaussianPrior, mean = 0.0, sigma = 0.3
            einstein_radius                      UniformPrior, lower_limit = 0.0, upper_limit = 8.0
    source
        redshift                                 1.0
        disk
            centre
                centre_0                         GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                         GaussianPrior, mean = 0.0, sigma = 0.3
            elliptical_comps
                elliptical_comps_0               GaussianPrior, mean = 0.0, sigma = 0.5
                elliptical_comps_1               GaussianPrior, mean = 0.0, sigma = 0.5
            intensity                            LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            effective_radius                     UniformPrior, lower_limit = 0.0, upper_limit = 30.0

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

The ``info`` attribute can be printed to give the results in a readable format:

.. code-block:: python

    print(result_list.info)

This gives the following output:

.. code-block:: bash

    Bayesian Evidence                              6333.47023932
    Maximum Log Likelihood                         6382.79198627
    Maximum Log Posterior                          1442056.41248673
    
    model                                          CollectionPriorModel (N=18)
        galaxies                                   CollectionPriorModel (N=18)
            lens                                   Galaxy (N=12)
                bulge                              EllSersic (N=7)
                mass                               EllIsothermal (N=5)
            source                                 Galaxy (N=6)
                disk                               EllExponential (N=6)
    
    Maximum Log Likelihood Model:
    
    galaxies
        lens
            bulge
                centre
                    centre_0                       0.369
                    centre_1                       -0.169
                elliptical_comps
                    elliptical_comps_0             0.766
                    elliptical_comps_1             0.061
                intensity                          0.000
                effective_radius                   1.161
                sersic_index                       1.597
            mass
                centre
                    centre_0                       -0.002
                    centre_1                       0.004
                elliptical_comps
                    elliptical_comps_0             -0.037
                    elliptical_comps_1             -0.107
                einstein_radius                    1.616
        source
            disk
                centre
                    centre_0                       -0.002
                    centre_1                       0.000
                elliptical_comps
                    elliptical_comps_0             0.165
                    elliptical_comps_1             -0.025
                intensity                          0.252
                effective_radius                   0.127
    
    
    Summary (3.0 sigma limits):
    
    galaxies
        lens
            bulge
                centre
                    centre_0                       0.0236 (-0.7006, 0.7200)
                    centre_1                       0.0218 (-0.6997, 1.0533)
                elliptical_comps
                    elliptical_comps_0             -0.0801 (-0.9960, 0.9758)
                    elliptical_comps_1             0.0775 (-0.9711, 0.9989)
                intensity                          0.0000 (0.0000, 0.0000)
                effective_radius                   11.2907 (0.0573, 29.6304)
                sersic_index                       2.7800 (0.8359, 4.9234)
            mass
                centre
                    centre_0                       -0.0036 (-0.0081, 0.0010)
                    centre_1                       0.0039 (-0.0003, 0.0087)
                elliptical_comps
                    elliptical_comps_0             -0.0368 (-0.0398, -0.0338)
                    elliptical_comps_1             -0.1079 (-0.1116, -0.1037)
                einstein_radius                    1.6160 (1.6129, 1.6195)
        source
            disk
                centre
                    centre_0                       -0.0024 (-0.0055, 0.0013)
                    centre_1                       0.0003 (-0.0033, 0.0037)
                elliptical_comps
                    elliptical_comps_0             0.1669 (0.1430, 0.2035)
                    elliptical_comps_1             -0.0244 (-0.0408, -0.0035)
                intensity                          0.2499 (0.2401, 0.2587)
                effective_radius                   0.1275 (0.1245, 0.1309)
    
    
    Summary (1.0 sigma limits):
    
    galaxies
        lens
            bulge
                centre
                    centre_0                       0.0236 (-0.2004, 0.2672)
                    centre_1                       0.0218 (-0.2204, 0.2282)
                elliptical_comps
                    elliptical_comps_0             -0.0801 (-0.4468, 0.2718)
                    elliptical_comps_1             0.0775 (-0.3457, 0.4478)
                intensity                          0.0000 (0.0000, 0.0000)
                effective_radius                   11.2907 (3.0980, 19.0891)
                sersic_index                       2.7800 (1.5561, 3.9258)
            mass
                centre
                    centre_0                       -0.0036 (-0.0051, -0.0021)
                    centre_1                       0.0039 (0.0026, 0.0057)
                elliptical_comps
                    elliptical_comps_0             -0.0368 (-0.0379, -0.0357)
                    elliptical_comps_1             -0.1079 (-0.1090, -0.1066)
                einstein_radius                    1.6160 (1.6149, 1.6170)
        source
            disk
                centre
                    centre_0                       -0.0024 (-0.0036, -0.0013)
                    centre_1                       0.0003 (-0.0009, 0.0016)
                elliptical_comps
                    elliptical_comps_0             0.1669 (0.1567, 0.1777)
                    elliptical_comps_1             -0.0244 (-0.0304, -0.0180)
                intensity                          0.2499 (0.2470, 0.2532)
                effective_radius                   0.1275 (0.1265, 0.1285)
    
    instances
    
    galaxies
        lens
            redshift                               0.5
        source
            redshift                               1.0

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

The script ``autolens_workspace/*/results`` contains a full description of all information contained
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

Linear Light Profiles
---------------------

**PyAutoLens** supports 'linear light profiles', where the ``intensity`` parameters of all parametric components are 
solved via linear algebra every time the model is fitted using a process called an inversion. This inversion always 
computes ``intensity`` values that give the best fit to the data (e.g. they maximize the likelihood) given the other 
parameter values of the light profile.

The ``intensity`` parameter of each light profile is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in the example below by 3) and removing 
the degeneracies that occur between the ``intnensity`` and other light profile
parameters (e.g. ``effective_radius``, ``sersic_index``).

For complex models, linear light profiles are a powerful way to simplify the parameter space to ensure the best-fit
model is inferred.

.. code-block:: python

    sersic_linear = al.lp_linear.EllSersic()
    
    lens_model_linear = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.EllDevVaucouleurs,
        disk=ag.lp_linear.EllSersic,
    )
    
    source_model_linear = af.Model(al.Galaxy, redshift=1.0, disk=al.lp_linear.EllExponential)

Wrap-Up
-------

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a non-linear search is and strategies to fit complex lens model to data in efficient and
robust ways.


