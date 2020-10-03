.. _modeling:

Lens Modeling
-------------

We can use a `Tracer` to fit ``data`` of a strong lens and use the `Tracer`'s model-image to quantify its goodness-of-fit.
Of course, when observe an image of a strong lens, we have no idea what *LightProfiles* and *MassProfiles* we should
give our `Tracer` to best reproduce the strong lens we observed:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/image.png
  :width: 400
  :alt: Alternative text

The task of finding the right *LightProfiles* and *MassProfiles* is called *lens modeling*. To begin, we must introduce
the *GalaxyModel* object, which behaves analogously to the `Galaxy` objects we've seen already. however, instead of
manually inputting the parameters of the `LightProfile`'s and `MassProfile`'s, for a *GalaxyModel* these are inferred
by fitting the strong lens data.

.. code-block:: bash

    lens_galaxy_model = al.GalaxyModel(
        redshift=0.5, light=al.lp.EllipticalDevVaucouleurs, mass=al.mp.EllipticalIsothermal
    )
    source_galaxy_model = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalExponential)

In the example above, we will fit our strong lens ``data`` two galaxies:

    - A lens galaxy with a *EllipticalDevVaucouleurs* `LightProfile` and *EllipticalIsothermal* `MassProfile`.
    - A source galaxy with a *EllipticalExponential* light profile.

The redshifts of the lens (z=0.5) and source(z=1.0) are fixed.

To perform the model-fit, we create a `PhaseImaging` object and 'run' the phase by passing it the *Imaging* dataset
and *Mask2D*.

We also pass it a *non-linear search*, which is the algorithm used to determine the set of `LightProfile` and
`MassProfile` parameters that best-fit our data, that is, that minimize the *residuals* and *chi-squared* values and
maximize its *log likelihood*.

.. code-block:: bash

    phase = al.PhaseImaging(
        galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
        phase_name="phase_example",
        search=af.DynestyStatic(),
    )

    result = phase.run(data=imaging, mask=mask)

The `PhaseImaging` object above returned a 'result', or ``Result`` object. This contains the maximum log likelihood
`Tracer` and *FitImaging*, which can easily be plotted.

.. code-block:: bash

    aplt.Tracer.subplot_tracer(
        tracer=result.max_log_likelihood_tracer, grid=mask.geometry.masked_grid
    )
    aplt.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

Here's what the model-fit of the model which maximizes the log likelihood looks like, providing good residuals and
low chi-squared values:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/subplot_fit.png
  :width: 600
  :alt: Alternative text

In fact, this ``Result`` object contains the full posterior information of our ``NonLinearSearch``, including all
parameter samples, log likelihood values and tools to compute the errors on the lens model. The autolens_workspace
contains a full description of all information contained in a ``Result``.

*GalaxyModel*'s can be fully customized, mkaing it simple to parameterize and fit many different lens models using any
combination of `LightProfile`'s and `MassProfile`'s light profiles:

.. code-block:: bash

    """
    This aligns the light and mass profile centres in the model, reducing the
    number of free parameter fitted for by Dynesty by 2.
    """

    lens_galaxy_model.light.centre = lens_galaxy_model.mass.centre

    """
    This fixes the lens galaxy light profile's effective radius to a value of
    0.8 arc-seconds, removing another free parameter.
    """

    lens_galaxy_model.light.effective_radius = 0.8

    """This forces the mass profile's einstein radius to be above 1.0 arc-seconds."""

    lens_galaxy_model.mass.einstein_radius > 1.0

The above fit used the *non-linear search* ``dynesty``, but **PyAutoLens** supports many other methods and their
setting can be easily customized:

.. code-block:: bash

    """Nested Samplers"""

    search = af.MultiNest(n_live_points=50, sampling_efficiency=0.5, evidence_tolerance=0.8)
    search = af.DynestyStatic(n_live_points=50, sample="rwalk")
    search = af.DynestyDynamic(sample="hslice")

    """MCMC"""

    search = af.Emcee(nwalkers=50, nsteps=500)

    """Optimizers"""

    search = af.PySwarmsLocal(n_particles=50)
    search = af.PySwarmsGlobal(n_particles=50).

Lens modeling with **PyAutoLens** is built around the probablstic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a *non-linear search* is and strategies to fit complex lens model to ``data`` in efficient and robust
way.


