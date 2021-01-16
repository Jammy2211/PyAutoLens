.. _fitting:

Fitting Data
------------

``Tracer`` objects represent a strong lensing system, allowing us to create an image of how the lens and source
``Galaxy``'s. Now, lets use a ``Tracer`` to fit ``Imaging`` ``data`` of a strong lens, which we begin by loading
from .fits files as an ``Imaging`` object:

.. code-block:: bash

    dataset_path = "/path/to/dataset/folder"

    imaging = al.Imaging.from_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    aplt.Imaging.image(imaging=imaging)
    aplt.Imaging.noise_map(imaging=imaging)
    aplt.Imaging.psf(imaging=imaging)

Here's what our image, ``noise_map`` and point-spread function look like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/noise_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/psf.png
  :width: 400
  :alt: Alternative text

We now need to mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit. To do
this we can use a ``Mask2D`` object, which for this example we'll create as a 3.0" circle.

.. code-block:: bash

    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    aplt.Imaging.image(imaging=masked_imaging)

Here is what our image looks like with the mask applied, where **PyAutoLens** has automatically zoomed around the
``Mask2D`` to make the lensed source appear bigger:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/masked_image.png
  :width: 400
  :alt: Alternative text

Following the lensing API guide, we can make a ``Tracer`` from a collection of ``LightProfile``, ``MassProfile`` and
``Galaxy`` objects. This ``Tracer`` then allows us to create an image of the strong lens system.

By passing a ``Tracer`` and ``MaskImaging`` object to a ``FitImaging`` object, we create a model-image from the ``Tracer``.
The model-image is the image of the ``Tracer`` blurred with the ``Imaging`` dataset's PSF, ensuring our fit to the data
provides a like-with-like comparison.

.. code-block:: bash

    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
fit_imaging_plotter.figures(model_image=True)

Here is how the ``Tracer``'s image and the ``FitImaging``'s model-image look; note how the model-image has been blurred
with the PSF of our dataset:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/tracer_image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/model_image.png
  :width: 400
  :alt: Alternative text

The ``FitImaging`` object does a lot more than just create the model-image, it also subtracts this image from
the ``data`` to produce a residual-map and weight these residuals by the noise to compute a chi-squared-map, both of which we can plot:

.. code-block:: bash

    aplt.FitImaging.residual_map(fit=fit)
    aplt.FitImaging.chi_squared_map(fit=fit)

For a good lens model where the ``Tracer``'s model image is representative of the strong lens system the residuals and
chi-squared values minimized:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/residual_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/chi_squared_map.png
  :width: 400
  :alt: Alternative text

In contrast, a bad lens model will show features in the residual-map and chi-squareds:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/bad_residual_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/fitting/bad_chi_squared_map.png
  :width: 400
  :alt: Alternative text

Most importantly, the ``FitImaging`` object also provides us with a log likelihood, a single value measure of how good
our ``Tracer`` fitted the dataset. If we can find a ``Tracer`` that produces a high log likelihood, we'll have a model
which is representative of our strong lens data! This task, called lens modeling, is covered in the next API overview.

Given a strong lens dataset, how do we determine a 'good' lens model? How do we determine the tracer (and therefore
combination of light profiles, mass profiles and galaxies) that minimize the residuals and chi-squared values?

This requires lens modeling, which uses a ``NonLinearSearch`` to fit many different tracers to the data.
This model-fitting is handled by our project **PyAutoFit**, a probabilistic programming language for non-linear model
fitting. Below, we setup our model as ``GalaxyModel`` objects, which repesent the galaxies we fit to our data:

If you are unfamilar ``data`` and model fitting, and unsure what terms like 'residuals', 'chi-sqaured' or 'likelihood' mean,
we'll explain all in chapter 1 of the **HowToLens** lecture series. Checkout the
`tutorials <https://pyautolens.readthedocs.io/en/latest/tutorials/howtolens.html>`_ section of the readthedocs!