.. _overview_3_modeling:

Lens Modeling
=============

We can use a ``Tracer`` to fit data of a strong lens and quantify its goodness-of-fit via a
*log_likelihood*.

Of course, when observe an image of a strong lens, we have no idea what combination of
``LightProfile``'s and ``MassProfiles``'s will produce a model-image that looks like the strong lens we observed:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/fitting/image.png
  :width: 400
  :alt: Alternative text

The task of finding these ``LightProfiles``'s and ``MassProfiles``'s is called *lens modeling*.

PyAutoFit
---------

Lens modelingtick_maker.min_value uses the probabilistic programming language
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

    # Lens:

    bulge = af.Model(al.lp.DevVaucouleurs)
    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass
    )

    # Source:

    disk = af.Model(al.lp.Exponential)

    source = af.Model(al.Galaxy, redshift=1.0, disk=disk)

We combine the lens and source model galaxies above into a ``Collection``, which is the model we will fit. Note how
we could easily extend this object to compose highly complex models containing many galaxies.

The reason we create separate ``Collection``'s for the ``galaxies`` and ``model`` is because the `model`
can be extended to include other components than just galaxies.


In this example, we therefore fit our strong lens data with two galaxies:

    - A lens galaxy with a elliptical Dev Vaucouleurs ``LightProfile`` representing a bulge and
      elliptical isothermal ``MassProfile`` representing its mass.
    - A source galaxy with an elliptical exponential ``LightProfile`` representing a disk.

The redshifts of the lens (z=0.5) and source(z=1.0) are fixed.

Printing the ``info`` attribute of the model shows us this is the model we are fitting, and shows us the free parameters and
their priors:

.. code-block:: python

    # Overall Lens Model:

    galaxies = af.Collection(lens=lens, source=source)
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

    analysis = al.AnalysisImaging(dataset=dataset)

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

This is contained in the ``Samples`` object. Below, we show how to print the median PDF parameter estimates, but
many different results are available and illustrated in the `results package of the workspace <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/results>`_.

.. code-block:: python

    samples = result.samples

    median_pdf_instance = samples.median_pdf()

    print("Median PDF Model Instances: \n")
    print(median_pdf_instance, "\n")
    print(median_pdf_instance.galaxies.galaxy.bulge)
    print()

This result contains the full posterior information of our non-linear search, including all
parameter samples, log likelihood values and tools to compute the errors on the lens model.

**PyAutoLens** includes many visualization tools for plotting the results of a non-linear search, for example we can
make a corner plot of the probability density function (PDF):

.. code-block:: python

    search_plotter = aplt.DynestyPlotter(samples=result.samples)
    search_plotter.cornerplot()

Here is an example of how a PDF estimated for a lens model appears:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/modeling/cornerplot.png
  :width: 600
  :alt: Alternative text

The result also contains the maximum log likelihood ``Tracer`` and ``FitImaging`` objects and which can easily be
plotted.

.. code-block:: python

    tracer_plotter = aplt.TracerPlotter(tracer=result.max_log_likelihood_tracer, grid=mask.derive_grid.masked)
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

Here's what the model-fit of the model which maximizes the log likelihood looks like, providing good residuals and
low chi-squared values:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/fitting/subplot_fit.png
  :width: 600
  :alt: Alternative text

The script ``autolens_workspace/*/results`` contains a full description of all information contained
in a ``Result``.

Model Customization
-------------------

The ``Model`` can be fully customized, making it simple to parameterize and fit many different lens models
using any combination of ``LightProfile``'s and ``MassProfile``'s:

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

    # Lens:

    bulge = af.Model(al.lp_linear.DevVaucouleurs)
    disk = af.Model(al.lp_linear.Sersic)
    
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        disk=disk
    )

    # Source:

    disk = af.Model(al.lp_linear.Exponential)

    source = af.Model(al.Galaxy, redshift=1.0, disk=disk)

Multi Gaussian Expansion
------------------------

A natural extension of linear light profiles are basis functions, which group many linear light profiles together in
order to capture complex and irregular structures in a galaxy's emission.

Using a clever model parameterization a basis can be composed which corresponds to just N = 5-10 parameters, making
model-fitting efficient and robust.

Below, we compose a basis of 30 Gaussians which all share the same `centre` and `ell_comps`. Their `sigma`
values are set in growing log10 bins in steps from 0.01 to 3.0 arc-seconds, which is the radius of the mask.

The `Basis` objects below can capture very complex light distributions with just N = 4 non-linear parameters!

.. code-block:: python

    total_gaussians = 30

    # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
    log10_sigma_list = np.linspace(-2, np.log10(3.0), total_gaussians)

    # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    # A list of Gaussian model components whose parameters are customized belows.

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # Iterate over every Gaussian and customize its parameters.

    for i, gaussian in enumerate(gaussian_list):

        # All Gaussians have same y and x centre.

        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1

        # All Gaussians have same elliptical components.

        gaussian.ell_comps = gaussian_list[0].ell_comps

        # All Gaussian sigmas are fixed to values above.

        gaussian.sigma = 10 ** log10_sigma_list[i]

    # The Basis object groups many light profiles together into a single model component.

    bulge = af.Model(
        al.lp_basis.Basis,
        light_profile_list=gaussian_list,
    )

The bulge's ``info`` attribute describes the basis model composition:

.. code-block:: python

    print(bulge.info)

Below is a snippet of the model, showing that different Gaussians are in the model parameterization:

.. code-block:: bash

    Total Free Parameters = 6

    model                                                                           Basis (N=6)
        light_profile_list                                                          Collection (N=6)
            0                                                                       Gaussian (N=6)
                sigma                                                               SumPrior (N=2)
                    other                                                           MultiplePrior (N=1)
            1                                                                       Gaussian (N=6)
                sigma                                                               SumPrior (N=2)
                    other                                                           MultiplePrior (N=1)
            2                                                                       Gaussian (N=6)
            ...
            trimmed for conciseness
            ...


    light_profile_list
        0
            centre
                centre_0                                                            GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                                                            GaussianPrior, mean = 0.0, sigma = 0.3
            ell_comps
                ell_comps_0                                                  GaussianPrior, mean = 0.0, sigma = 0.3
                ell_comps_1                                                  GaussianPrior, mean = 0.0, sigma = 0.3
            sigma
                bulge_a                                                             UniformPrior, lower_limit = 0.0, upper_limit = 0.2
                other
                    bulge_b                                                         UniformPrior, lower_limit = 0.0, upper_limit = 10.0
                    other                                                           0.0
        1
            centre
                centre_0                                                            GaussianPrior, mean = 0.0, sigma = 0.3
                centre_1                                                            GaussianPrior, mean = 0.0, sigma = 0.3
            ell_comps
                ell_comps_0                                                  GaussianPrior, mean = 0.0, sigma = 0.3
                ell_comps_1                                                  GaussianPrior, mean = 0.0, sigma = 0.3
            sigma
                bulge_a                                                             UniformPrior, lower_limit = 0.0, upper_limit = 0.2
                other
                    bulge_b                                                         UniformPrior, lower_limit = 0.0, upper_limit = 10.0
                    other                                                           0.3010299956639812
        2
        ...
        trimmed for conciseness
        ...

Shapelets
---------

**PyAutoLens** also supports Shapelet basis functions, which are appropriate for capturing exponential / disk-like
features in a galaxy, and therefore may make a good model for most lensed source galaxies.

This is illustrated in full on the ``autogalaxy_workspace`` in the example
script autolens_workspace/scripts/imaging/modeling/advanced/shapelets.py .

Regularization
--------------

**PyAutoLens** can also apply Bayesian regularization to Basis functions, which smooths the linear light profiles
(e.g. the Gaussians) in order to prevent over-fitting noise.

.. code-block:: python

    bulge = af.Model(
        al.lp_basis.Basis, light_profile_list=gaussians_lens, regularization=al.reg.Constant
    )

Wrap-Up
-------

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a non-linear search is and strategies to fit complex lens model to data in efficient and
robust ways.


