.. _pixelizations:

Pixelized Source Reconstruction
-------------------------------

**PyAutoLens** can reconstruct the light of the strong lensed source-galaxy using a pixel-grid, using a process
called an ``Inversion``.

Lets use a ``Pixelization`` to reconstruct the source-galaxy of the image below, noting how complex the lensed source
appears, with multiple rings and clumps of light:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/image.png
  :width: 400
  :alt: Alternative text

We are going to fit this image with an ``Inversion``, so we first create ``Mask2D`` and ``MaskImaging`` objects:

.. code-block:: bash

   mask = al.Mask2D.circular(
      shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.6
    )

   masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

To reconstruct the source using a pixel-grid, we simply pass it the ``Pixelization`` class we want to reconstruct its
light using. We also pass a ``Regularization`` scheme which describes our prior on how much we smooth the reconstruction.

Below, we use a ``Rectangular`` pixelization with resolution 40 x 40 and ``Constant`` regularizaton scheme:

.. code-block:: bash

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.Rectangular(shape=(40, 40)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

To fit the data, we simply pass this source-galaxy into a ``Tracer`` (complete with lens galaxy mass model). The
``FitImaging`` object will automatically use the source galaxy's ``Pixelization`` and ``Regularization`` to reconstruct
the lensed source's light using the ``Inversion``:

.. code-block:: bash

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

Here is what our reconstructed source galaxy looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/rectangular.png
  :width: 400
  :alt: Alternative text

Note how the source reconstruction is irregular and has multiple clumps of light - these features would be difficult
to represent using ``LightProfile``'s!

The source reconstruction can be mapped back to the image-plane, to produce a reconstructed image:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/reconstructed_image.png
  :width: 400
  :alt: Alternative text

**PyAutoLens** supports many different pixel-grids. Below, we use a *VoronoiMagnification* pixelization, which defines
the source-pixel centres in the image-plane and ray traces them to the source-plane.

The source pixel-grid is therefore adapted to the mass-model magnification pattern, placing more source-pixel in the
highly magnified regions of the source-plane.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/voronoi.png
  :width: 400
  :alt: Alternative text

By inspecting the residual-map, normalized residual-map and chi-squared-map of the ``FitImaging`` object, we can see how
the source reconstruction accurately fits the image of the strong lens:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/pixelizations/voronoi_fit.png
  :width: 600
  :alt: Alternative text

This was a brief overview of *Inversions* with **PyAutoLens**. There is a lot more to using *Inversions* then presented
here, which is covered in chapters 4 and 5 of the **HowToLens**, specifically:

    - How the source reconstruction determines the flux-values of the source it reconstructs.
    - The Bayesian framework employed to choose the approrpriate level of ``Regularization`` and avoid overfitting noise.
    - Unphysical lens model solutions that often arise when using an ``Inversion``.
    - Advanced ``Pixelization`` and ``Regularization`` schemes that adapt to the source galaxy being reconstructed.