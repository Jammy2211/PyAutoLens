.. _simulate:

Simulating Lenses
-----------------

**PyAutoLens** provides tool for simulating strong lens data-sets, which can be used to test lens modeling pipelines
and train neural networks to recognise and analyse images of strong lenses.

Simulating strong lens images uses a *SimulatorImaging* object, which models the process that an instrument like the
Hubble Space Telescope goes through observe a strong lens. This includes accounting for the exposure time to
determine the signal-to-noise of the data, blurring the observed light of the strong lens with the telescope optics
and accounting for the background sky in the exposure which adds Poisson noise:

.. code-block:: bash

    psf = al.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        background_sky_level=1.0,
        psf=psf,
        add_poisson_noise=True,
    )

Once we have a simulator, we can use it to create an imaging dataset which consists of an image, noise-map and
Point Spread Function (PSF) by passing it a tracer and grid:

.. code-block:: bash

    imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

Here is what our dataset looks like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/noise_map.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/psf.png
  :width: 400
  :alt: Alternative text

The `autolens_workspace` includes many example simulators for simulating strong lenses with a range of different
physical properties and for creating imaging datasets for a variety of telescopes (e.g. Hubble, Euclid).

Below, we show what a strong lens looks like for different instruments.

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/vro_image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/euclid_image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/hst_image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/simulating/ao_image.png
  :width: 400
  :alt: Alternative text