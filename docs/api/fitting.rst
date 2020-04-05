.. _api:

API - Fitting
-------------

**PyAutoLens** can create *Tracer* objects to represent a strong lensing system. Now, we're going use these objects to
fit imaging data of a strong lens, which we begin by loading from .fits files:

.. code-block:: bash

    dataset_path = "/path/to/dataset/folder"

    imaging = al.Imaging.from_fits(
        image_path=dataset_path + "image.fits",
        psf_path=dataset_path + "psf.fits",
        noise_map_path=dataset_path + "noise_map.fits",
        pixel_scales=0.1,
    )

    aplt.Imaging.image(imaging=imaging)
    aplt.Imaging.noise_map(imaging=imaging)
    aplt.Imaging.psf(imaging=imaging)

Here's what our image, noise-map and point-spread function look like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/noise_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/psf.png
  :width: 400
  :alt: Alternative text

We now need to mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit:

.. code-block:: bash

    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    aplt.Imaging.image(imaging=masked_imaging)

Here is what our image looks like with the mask applied, where PyAutoLens has automatically zoomed around the mask
to make the lensed source appear bigger:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/masked_image.png
  :width: 400
  :alt: Alternative text

Following the lensing API guide, we can make a tracer from a collection of *LightProfile*, *MassProfile* and *Galaxy*
objects. We can then use the *FitImaging* object to fit this tracer to the dataset, performing all necessary tasks
to create the model image we fit the data with, such as blurring the tracer's image with the imaging PSF:

.. code-block:: bash

    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

    aplt.FitImaging.model_imagefit=fit)
    aplt.FitImaging.residual_map(fit=fit)
    aplt.FitImaging.chi_squared_map(fit=fit)

For a good lens model where the model image and tracer are representative of the strong lens system the
residuals and chi-squared values minimized:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/residual_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/chi_squared_map.png
  :width: 400
  :alt: Alternative text

In contrast, a bad lens model will show features in the residual-map and chi-squareds:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/bad_residual_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/fitting/bad_chi_squared_map.png
  :width: 400
  :alt: Alternative text

Given a strong lens dataset, how do we determine a 'good' lens model? How do we determine the tracer (and therefore
combination of light profiles, mass profiles and galaxies) that minimize the residuals and chi-squared values?

This requires lens modeling, which uses a non-linear search algorithm to fit many different tracers to the data.
This model-fitting is handled by our project **PyAutoFit**, a probablistic programming language for non-linear model
fitting. Below, we setup our model as *GalaxyModel* objects, which repesent the galaxies we fit to our data:

.. code-block:: bash

    lens_galaxy_model = al.GalaxyModel(
        redshift=0.5, light=al.lp.EllipticalDevVaucouleurs, mass=al.mp.EllipticalIsothermal
    )
    source_galaxy_model = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalExponential)

This means we will fit our data with two galaxies, a lens and source galaxy, with the light and mass profiles input
into the *GalaxyModel* objects.

To perform the fit, we create a *PhaseImaging* object and 'run' the phase by passing it the dataset and mask. We also
pass it a non-linear search class, which instructs the phase to fit the lens data using the algorithm **PyMultiNest**.

.. code-block:: bash

    phase = al.PhaseImaging(
        galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
        phase_name="phase_example",
        non_linear_class=af.MultiNest,
    )

    phase.run(data=imaging, mask=mask)

By changing the *GalaxyModel* objects it is simple to parameterize and fit many different lens models using different
combinations of light profiles, mass profiles and perhaps even modeling the system with different numbers of galaxies!

**PyAutoFit** provides us with many ways to customize our model fit.

.. code-block:: bash

    # This aligns the light and mass profile centres in the model, reducing the number of free parameter fitted for by
    # MultiNest by 2.
    lens_galaxy_model.light.centre = lens_galaxy_model.mass.centre

    # This fixes the lens galaxy light profile's rotation angle phi to a value of 45.0 degrees, removing another
    # free parameter.
    lens_galaxy_model.light.phi = 45.0

    # This forces the mass profile to be rounder than the light profile.
    lens_galaxy_model.mass.axis_ratio > lens_galaxy_model.light.axis_ratio

There is a lot more to lens modeling with **PyAutoLens** than shown here. For example, to fit complex lens models we
use *Pipeline* objects, that chain together a series of the phase fits shown above. The pipeline changes the lens model
between phases, using the fits of earlier phases to guide the non-linear search in later phases.

You can learn more about advanced lens modeling in **PyAutoens** in chapters 2 and 3 of the **HowToLens** lecture series.

**PyAutoLens** also allows on to reconstruct the lensed source galaxy's light on a pixel-grid. This is important for
modeling real galaxies, whose appear are typically irregular with non-symmetric features such spiral arms and clumps of
star formation.

Using pixelized sources is simply, we simply input them into our *Galaxy* or *GalaxyModel* objects:

.. code-block:: bash

    source_galaxy_model = al.GalaxyModel(redshift=1.0,
                                         pixelization=al.pix.VoronoiMagnification,
                                         regularization=al.reg.Constant)

Heres how a real strong lens's reconstructed source appears on a Voronoi pixel-grid:

