.. _modeling:

Lens Modeling
-------------

We can use a ``Tracer`` to fit ``data`` of a strong lens and use the ``Tracer``'s model-image to quantify its
goodness-of-fit. Of course, when observe an image of a strong lens, we have no idea what ``LightProfile``'s and
``MassProfiles``'s we should give our ``Tracer`` to best reproduce the strong lens we observed:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/image.png
  :width: 400
  :alt: Alternative text

The task of finding the right ``LightProfiles``'s and ``MassProfiles``'s is called *lens modeling*.

Lens modeling with **PyAutoLens** uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import it separately to **PyAutoLens**

.. code-block:: bash

    import autofit as af

We compose the lens model that we fit to the data using ``GalaxyModel`` objects. These behave analogously to ``Galaxy``
objects but their  ``LightProfile`` and ``MassProfile`` parameters are not specified and are instead determined by a
fitting procedure.

.. code-block:: bash

    lens_galaxy_model = al.GalaxyModel(
        redshift=0.5,
        bulge=al.lp.EllipticalDevVaucouleurs,
        mass=al.mp.EllipticalIsothermal
    )
    source_galaxy_model = al.GalaxyModel(redshift=1.0, disk=al.lp.EllipticalExponential)

In the example above, we will fit our strong lens data with two galaxies:

    - A lens galaxy with a ``EllipticalDevVaucouleurs`` ``LightProfile`` representing a bulge and
      ``EllipticalIsothermal`` ``MassProfile`` representing its mass.
    - A source galaxy with a ``EllipticalExponential`` ``LightProfile`` representing a disk.

The redshifts of the lens (z=0.5) and source(z=1.0) are fixed.

We now choose the ``NonLinearSearch``, which is the fitting method used to determine the set of `LightProfile`
and `MassProfile` parameters that best-fit our data by minimizing the *residuals* and *chi-squared* values and
maximizing  its *log likelihood*.

In this example we use `dynesty` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm we find is
very effective at lens modeling.

.. code-block:: bash

    search = af.DynestyStatic(name="phase_example")

To perform the model-fit, we create a ``PhaseImaging`` object and 'run' the phase by passing it the dataset and mask.

.. code-block:: bash

    phase = al.PhaseImaging(
        search=search,
        galaxies=af.CollectionPriorModel(lens=lens_galaxy_model, source=source_galaxy_model),
    )

    result = phase.run(data=imaging, mask=mask)

The `NonLinearSearch` fits the lens model by guessing many lens models over and over iteratively, using the models which
give a good fit to the data to guide it where to guess subsequent model. An animation of a `NonLinearSearch` is shown
below,  where initial lens models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://media.githubusercontent.com/media/Jammy2211/autolens_files/main/lensmodel.gif
  :width: 600

The ``PhaseImaging`` object above returns a ``Result`` object, which contains the maximum log likelihood ``Tracer``
and ``FitImaging`` objects and which can easily be plotted.

.. code-block:: bash

    tracer_plotter = aplt.TracerPlotter(tracer=result.max_log_likelihood_tracer, grid=mask.masked_grid)
    tracer_plotter.subplot_tracer()

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_imaging_plotter.subplot_fit_imaging()

Here's what the model-fit of the model which maximizes the log likelihood looks like, providing good residuals and
low chi-squared values:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/subplot_fit.png
  :width: 600
  :alt: Alternative text

In fact, this ``Result`` object contains the full posterior information of our ``NonLinearSearch``, including all
parameter samples, log likelihood values and tools to compute the errors on the lens model.

The script ``autolens_workspace/examples/mdoel/result.py`` contains a full description of all information contained
in a ``Result``.

``GalaxyModel``'s can be fully customized, making it simple to parameterize and fit many different lens models using
any combination of ``LightProfile``'s and ``MassProfile``'s light profiles:

.. code-block:: bash

    lens_galaxy_model = al.GalaxyModel(
        redshift=0.5,
        bulge=al.lp.EllipticalDevVaucouleurs,
        mass=al.mp.EllipticalIsothermal
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

    """This forces the mass profile's einstein radius to be above 1.0 arc-seconds."""

    lens_galaxy_model.mass.einstein_radius > 1.0

The above fit used the `NonLinearSearch` ``dynesty``, but **PyAutoLens** supports many other methods and their
setting can be easily customized:

.. code-block:: bash

    """Nested Samplers"""

    search = af.MultiNest(name="multinest", n_live_points=50, sampling_efficiency=0.5, evidence_tolerance=0.8)
    search = af.DynestyStatic(name="dynesty_static", n_live_points=50, sample="rwalk")
    search = af.DynestyDynamic(name="dynesty_dynamic", sample="hslice")

    """MCMC"""

    search = af.Emcee(name="emcee", nwalkers=50, nsteps=500)

    """Optimizers"""

    search = af.PySwarmsLocal(name="pso_local", n_particles=50)
    search = af.PySwarmsGlobal(name="pso_global", n_particles=50).

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a ``NonLinearSearch`` is and strategies to fit complex lens model to data in efficient and
robust ways.


