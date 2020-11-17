.. _simulate:

Simulating Lenses
-----------------

**PyAutoLens** provides tool for simulating strong lens data-sets, which can be used to test lens modeling pipelines
and train neural networks to recognise and analyse images of strong lenses.

Simulating strong lenses begins by creating a *SimulatorImaging* object, which represents how a image is acquired and
processed when on a telescope CCD. This includes accounting for the exposure time in determine the signal to noise,
blurring the ``data`` due to the telescope optics, the background sky during taking the exposure and noise due to Poisson
counts of the signal:

.. code-block:: bash

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        background_sky_level=1.0,
        psf=psf,
        add_poisson_noise=True,
    )

Once we have a simulator, we can use it to create an imaging dataset (an image, noise-map, PSF) by pasing it a tracer
and grid:

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

The ``autolens_workspace`` includes example simulators for various existing and upcoming telescopes, for example the
Vera Rubin Observatry, Euclid, the Hubble Space Telescope and Keck Adaptive Optics Imaging. Below, we show what the
image above looks like for these different instruments.

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