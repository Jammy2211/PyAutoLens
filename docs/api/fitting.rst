.. _api:

API - Fitting
-------------

We've shown how with **PyAutoLens** we can create *Tracer* objects that represent a strong lensing system. Now, we're
going to look at how we can use these objects to fit imaging data of a strong lens, which we begin by loading from .fits
files:

.. code-block:: bash

    dataset_path = "/path/to/dataset/folder"

    imaging = al.Imaging.from_fits(
        image_path=dataset_path + "image.fits",
        psf_path=dataset_path + "psf.fits",
        noise_map_path=dataset_path + "noise_map.fits",
        pixel_scales=0.1,
    )

    aplt.imaging.image(imaging=imaging)
    aplt.imaging.noise_map(imaging=imaging)
    aplt.imaging.psf(imaging=imaging, plotter=plotter)

Here's what our image, noise-map and point-spread function look like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/grid.png
  :width: 400
  :alt: Alternative text

We now need a mask that we will apply to the data, such that regions where there is no signal (e.g. the signal) are
omitted from the fit:

.. code-block:: bash

    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    aplt.imaging.image(imaging=masked_imaging)

Here is what our image looks like with the mask applied:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/grid.png
  :width: 400
  :alt: Alternative text

Following the lensing API guide, we can make a tracer from a collection of *LightProfile*, *MassProfile* and *Galaxy*
objects. We can then use the *FitImaging* module to fit this tracer to the data-set, performing all necessary tasks
to create the model imag we fit the data with, such as blurring the tracer's image with the imaging PSF:

.. code-block:: bash

    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

    aplt.fit_imaging.model_imagefit=fit)
    aplt.fit_imaging.residual_map(fit=fit)
    aplt.fit_imaging.chi_squared_map(fit=fit)

For a good lens model, that is one whose model image (and therefore tracer) is representative of the data one will
see the residuals and chi-squared values minimized:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/grid.png
  :width: 400
  :alt: Alternative text

In contrast, a poor lens model will show features in the residual-map and chi-squareds:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/images/grid.png
  :width: 400
  :alt: Alternative text

Of course, given a dataset, the quesiton is next how do we determine a 'good' lens model? How do we figure out an
appropriate tracer (and therefore combination of light profiles, mass profiles and galaxies) to minimize the residuals
and chi-squared values?

To do this, we need to perform lens modeling, which essentially fits many tracers to the data using a non-linear search
algorithm. This side of the model fitting is handled by our sister project **PyAutoFit**, which is a probablistic
programming language for non-linear model fitting. In the code below, we create a *PhaseImaging* object out of the
