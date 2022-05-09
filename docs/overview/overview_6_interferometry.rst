.. _overview_6_interferometry:

Interferometry
==============

Alongside CCD imaging data, **PyAutoLens** supports the modeling of interferometer data from submillimeter and radio
observatories.

The dataset is fitted directly in the uv-plane, circumventing issues that arise when fitting a 'dirty image' such as
correlated noise.

Real Space Mask
---------------

To begin, we define a real-space mask. Although interferometer lens modeling is performed in the uv-plane and
therefore Fourier space, we still need to define the grid of coordinates in real-space from which the lensed source's
images are computed. It is this image that is mapped to Fourier space to compare to the uv-plane data.

.. code-block:: python

    real_space_mask_2d = ag.Mask2D.circular(
        shape_native=(400, 400), pixel_scales=0.025, radius=3.0
    )

Interferometer Data
-------------------

We next load an ``Interferometer`` dataset from fits files, which follows the same API that we have seen
for an ``Imaging`` object.

.. code-block:: python

    dataset_path = "/path/to/dataset/folder"

    interferometer = al.Interferometer.from_fits(
        visibilities_path=path.join(dataset_path, "visibilities.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask_2d
    )

    interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
    interferometer_plotter.figures_2d(visibilities=True, uv_wavelengths=True)

Here is what the interferometer visibilities and uv wavelength (which represent the interferometer's baselines):

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/visibilities.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/uv_wavelengths.png
  :width: 400
  :alt: Alternative text

The data used in this overview contains only ~300 visibilities and is representative of a low resolution
Square-Mile Array (SMA) dataset.

We discuss below how **PyAutoLens** can scale up to large visibilities datasets from an instrument like ALMA.

This can also plot the dataset in real-space, using the fast Fourier transforms described below.

.. code-block:: python

    interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
    interferometer_plotter.figures_2d(dirty_image=True, dirty_signal_to_noise_map=True)

Here is what the image and signal-to-noise map look like in real space:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/dirty_image.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/dirty_signal_to_noise_map.png
  :width: 400
  :alt: Alternative text

UV-Plane FFT
------------

To perform uv-plane modeling, **PyAutoLens** Fourier transforms the lensed image (computed via a ``Tracer``) from
real-space to the uv-plane.

This operation uses a ``Transformer`` object, of which there are multiple available
in **PyAutoLens**. This includes a direct Fourier transform which performs the exact Fourier transform without approximation.

.. code-block:: python

    transformer_class = al.TransformerDFT

However, the direct Fourier transform is inefficient. For ~10 million visibilities, it requires thousands of seconds
to perform a single transform. This approach is therefore unfeasible for high quality ALMA and radio datasets.

For this reason, **PyAutoLens** supports the non-uniform fast fourier transform algorithm
**PyNUFFT** (https://github.com/jyhmiinlin/pynufft), which is significantly faster, being able too perform a Fourier
transform of ~10 million in less than a second!

.. code-block:: python

    transformer_class = al.TransformerNUFFT

To perform a fit, we follow the same process we did for imaging. We do not need to mask an interferometer dataset,
but we will apply the settings above:

.. code-block:: python

    interferometer = interferometer.apply_settings(
        settings=al.SettingsInterferometer(transformer_class=transformer_class)
    )

Fitting
-------

The interferometer can now be passed to a ``FitInterferometer`` object to fit it to a data-set:

.. code-block:: python

    fit = al.FitInterferometer(
        interferometer=interferometer, tracer=tracer
    )

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(fit=fit)
    fit_interferometer_plotter.subplot_fit_interferometer()
    fit_interferometer_plotter.subplot_fit_real_space()

Here is what the image of the tracer looks like before it is Fourier transformed to the uv-plane:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/image_pre_ft.png
  :width: 400
  :alt: Alternative text

And here is what the Fourier transformed model visibilities look like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/model_visibilities.png
  :width: 400
  :alt: Alternative text

Here is what the fit of the galaxy looks like in real space (which is computed via a FFT from the uv-plane):

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/fit_dirty_images.png
  :width: 400
  :alt: Alternative text

Pixelized Sources
-----------------

Interferometer data can also be modeled using pixelized source's, which again perform the source reconstruction by
directly fitting the visibilities in the uv-plane.

The source reconstruction is visualized in real space:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/reconstruction.png
  :width: 400
  :alt: Alternative text

Computing this source reconstruction would be extremely inefficient if **PyAutoLens** used a traditional approach to
linear algebra which explicitly stored in memory the values required to solve for the source fluxes. In fact, for an
interferometer dataset of ~10 million visibilities this would require **hundreds of GB of memory**!

**PyAutoLens** uses the library **PyLops** (https://pylops.readthedocs.io/en/latest/) to represent this calculation as
a sequence of memory-light linear operators.

The combination of **PyNUFFT** and **PyLops** makes the analysis of ~10 million visibilities from observatories such as
ALMA and JVLA feasible in **PyAutoLens**.

Lens Modeling
--------------

It is straight forward to fit a lens model to an interferometer dataset, using the same API that we saw for imaging
data in the modeling overview example.

Whereas we previously used an ``AnalysisImaging`` object, we instead use an ``AnalysisInterferometer`` object which fits
the lens model in the correct way for an interferometer dataset. This includes mapping the lens model from real-space
to the uv-plane via the Fourier transform discussed above:

.. code-block:: python

    lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
    source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, disk=al.lp.EllExponential)

    model = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)

    search = af.DynestyStatic(name="overview_interferometer")

    analysis = al.AnalysisInterferometer(dataset=interferometer)

    result = search.fit(model=model, analysis=analysis)

Simulations
-----------

Simulated interferometer datasets can be generated using the ``SimulatorInterferometer`` object, which includes adding
Gaussian noise to the visibilities:

.. code-block:: python

    real_space_grid_2d = ag.Grid2D.uniform(
        shape_native=real_space_mask.shape_native,
        pixel_scales=real_space_mask.pixel_scales
    )

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        background_sky_level=1.0,
        noise_sigma=0.01,
    )

    interferometer = simulator.via_tracer_from(tracer=tracer, grid=real_space_grid)

Wrap-Up
-------

The `interferometer <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/interferometer>`_ package
of the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ contains numerous example scripts for performing
interferometer modeling and simulating strong lens interferometer datasets.