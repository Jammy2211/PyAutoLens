.. _overview_5_pixelizations:

Pixelized Sources
=================

**PyAutoLens** can reconstruct the light of a strongly lensed source-galaxy using a pixel-grid, using a process
called an ``Inversion``.

Lets use a ``Pixelization`` to reconstruct the source-galaxy of the image below, noting how complex the lensed source
appears, with multiple rings and clumps of light:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/image.png
  :width: 400
  :alt: Alternative text

Rectangular Example
-------------------

To fit this image with an ``Inversion``, we first mask the ``Imaging`` object:

.. code-block:: python

   mask = al.Mask2D.circular(
      shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.6
    )

   imaging = imaging.apply_mask(mask=mask_2d)

To reconstruct the source using a pixel-grid, we simply pass it the ``Pixelization`` class we want to reconstruct its
light using.

We also pass a ``Regularization`` scheme which applies a smoothness prior on the source reconstruction.

Below, we use a ``Rectangular`` pixelization with resolution 40 x 40 and a ``Constant`` regularization scheme:

.. code-block:: python

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(shape=(40, 40)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

To fit the data, we simply pass this source-galaxy into a ``Tracer`` (complete with lens galaxy mass model). The
``FitImaging`` object will automatically use the source galaxy's ``Pixelization`` and ``Regularization`` to reconstruct
the lensed source's light using the ``Inversion``:

.. code-block:: python

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=imaging, tracer=tracer)

Here is what our reconstructed source galaxy looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/rectangular.png
  :width: 400
  :alt: Alternative text

Note how the source reconstruction is irregular and has multiple clumps of light, these features would be difficult
to represent using analytic light profiles!

The source reconstruction can be mapped back to the image-plane, to produce a reconstructed image:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/reconstructed_image.png
  :width: 400
  :alt: Alternative text

Voronoi Example
---------------

**PyAutoLens** supports many different pixel-grids. Below, we use a ``VoronoiMagnification`` pixelization, which
defines the source-pixel centres in the image-plane and ray traces them to the source-plane.

The source pixel-grid is therefore adapted to the mass-model magnification pattern, placing more source-pixel in the
highly magnified regions of the source-plane.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/voronoi.png
  :width: 400
  :alt: Alternative text

By inspecting the residual-map, normalized residual-map and chi-squared-map of the ``FitImaging`` object, we can see
how the source reconstruction accurately fits the image of the strong lens:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/voronoi_fit.png
  :width: 600
  :alt: Alternative text

Wrap-Up
-------

This was a brief overview of ``Inverion``'s with **PyAutoLens**.

There is a lot more to using ``Inverion``'s then presented here, which is covered in chapters 4 and 5 of
the **HowToLens**, specifically:

    - How the source reconstruction calculates the flux-values of the source pixels when it performs the reconsturction.
    - What exactly regularization is and why it is necessary.
    - The Bayesian framework employed to choose an appropriate level of smoothing and avoid overfitting noise.
    - How to perform lens modeling with inversions.
    - Advanced ``Pixelization`` and ``Regularization`` schemes that adapt to the source galaxy being reconstructed.