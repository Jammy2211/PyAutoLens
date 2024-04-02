.. _overview_6_interferometry:

Interferometry
==============

**PyAutoLens** supports modeling of interferometer data from submillimeter and radio observatories such as ALMA
or LOFAR.

The visibilities of the interferometer dataset are fitted directly in the uv-plane, circumventing issues that arise
when fitting a dirty image produced via the visibilities.

The most important issue this addresses is removing correlated noise from impacting the fit.

Real Space Mask
---------------

To begin, we define a real-space mask. Although interferometer lens modeling is performed in the uv-plane and
therefore Fourier space, we still need to define the grid of coordinates in real-space from which the lensed source's
images are computed. It is this image that is mapped to Fourier space to compare to the uv-plane data.

The size and resolution of this mask depend on the baselines of your interferometer dataset. datasets with longer
baselines (i.e. higher resolution data) require higher resolution and larger masks.

.. code-block:: python

    import autolens as al
    import autolens.plot as aplt

    real_space_mask_2d = ag.Mask2D.circular(
        shape_native=(400, 400), pixel_scales=0.025, radius=3.0
    )

Interferometer Data
-------------------

We next load an interferometer dataset from fits files, which follows the same API that we have seen for an ``Imaging``
object.

.. code-block:: python

    dataset_path = "/path/to/dataset/folder"

    dataset = al.Interferometer.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask_2d
    )

The **PyAutoLens** plot module has tools for plotting interferometer datasets, including the visibilities, noise-map
and uv wavelength which represent the interferometer's baselines.

The data used in this tutorial contains only ~300 visibilities and is representative of a low resolution
Square-Mile Array (SMA) dataset.

We made this choice so the script runs fast, and we discuss below how **PyAutoLens** can scale up to large visibilities
datasets from an instrument like ALMA.

.. code-block:: python

    dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
    dataset_plotter.figures_2d(data=True, uv_wavelengths=True)

Here is what the interferometer visibilities and uv wavelength (which represent the interferometer's baselines):

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/0_data.png?raw=true
  :width: 400
  :alt: Alternative text

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/1_uv_wavelengths.png?raw=true
  :width: 400
  :alt: Alternative text

It can also plot dirty images of the dataset in real-space, using the fast Fourier transforms described below.

.. code-block:: python

    dataset_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
    dataset_plotter.figures_2d(dirty_image=True, dirty_signal_to_noise_map=True)

Here is what the image and signal-to-noise map look like in real space:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/2_dirty_image.png?raw=true
  :width: 400
  :alt: Alternative text

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/3_dirty_signal_to_noise_map.png?raw=true
  :width: 400
  :alt: Alternative text

Tracer
------

To perform uv-plane modeling, **PyAutoLens** generates an image of the strong lens system in real-space via a tracer.

Lets quickly set up the ``Tracer`` we'll use in this example.

.. code-block:: python

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer, grid=real_space_mask.derive_grid.unmasked
    )
    tracer_plotter.figures_2d(image=True)

Here is what the image of the tracer looks like:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/4_image_2d.png?raw=true
  :width: 600
  :alt: Alternative text

UV-Plane FFT
------------

To perform uv-plane modeling, **PyAutoLens** next Fourier transforms this image from real-space to the uv-plane.

This operation uses a ``Transformer`` object, of which there are multiple available
in **PyAutoLens**. This includes a direct Fourier transform which performs the exact Fourier transform without approximation.

.. code-block:: python

    transformer_class = al.TransformerDFT

However, the direct Fourier transform is inefficient. For ~10 million visibilities, it requires thousands of seconds
to perform a single transform. This approach is therefore unfeasible for high quality ALMA and radio datasets.

For this reason, **PyAutoLens** supports the non-uniform fast fourier transform algorithm
**PyNUFFT** (https://github.com/jyhmiinlin/pynufft), which is significantly faster, being able to perform a Fourier
transform of ~10 million in less than a second!

.. code-block:: python

    transformer_class = al.TransformerNUFFT

To use this transformer in a fit, we use the ``apply_settings`` method.

.. code-block:: python

    dataset = dataset.apply_settings(
        settings=al.SettingsInterferometer(transformer_class=transformer_class)
    )

Fitting
-------

The interferometer can now be passed to a ``FitInterferometer`` object to fit it to a dataset:

.. code-block:: python

    fit = al.FitInterferometer(
        interferometer=interferometer, tracer=tracer
    )


Visualization of the fit is provided both in the uv-plane and in real-space.

Note that the fit is not performed in real-space, but plotting it in real-space is often more informative.

.. code-block:: python

    fit = al.FitInterferometer(
        interferometer=interferometer, tracer=tracer
    )

    fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
    fit_plotter.subplot_fit()

Here is what the subplot image looks like:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/5_subplot_fit.png?raw=true
  :width: 800
  :alt: Alternative text

Pixelized Sources
-----------------

Interferometer data can also be modeled using pixelized source's, which again performs the source reconstruction by
directly fitting the visibilities in the uv-plane.

.. code-block:: python

    pixelization = al.Pixelization(
            image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

The source reconstruction is visualized in real space:

.. code-block:: python

    fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
    fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

Here is what it looks like:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/6_reconstruction.png?raw=true
  :width: 600
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
data.

We first compose the model, omitted the lens light components given that most strong lenses observed at submm /
radio wavelengths do not have visible lens galaxy emission.

.. code-block:: python

    # Lens:

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    # Source:

    disk = af.Model(al.lp.Exponential)

    source = af.Model(al.Galaxy, redshift=1.0, disk=disk)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

We again choose the non-linear search ``dynesty`` (https://github.com/joshspeagle/dynesty).

.. code-block:: python

    search = af.Nautilus(path_prefix="overview", name="interferometer")

Whereas we previously used an ``AnalysisImaging`` object, we instead use an ``AnalysisInterferometer`` object which fits
the lens model in the correct way for an interferometer dataset.

This includes mapping the lens model from real-space to the uv-plane via the Fourier transform discussed above.

.. code-block:: python

    analysis = al.AnalysisInterferometer(dataset=dataset)

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

The results can be found in the ``output/overview_interferometer`` folder in the ``autolens_workspace``.

.. code-block:: python

    result = search.fit(model=model, analysis=analysis)

The **PyAutoLens** visualization library and ``FitInterferometer`` object includes specific methods for plotting the
results, for example the maximum log likelihood fit:

.. code-block:: python

    fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

Here is what it looks like:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/7_subplot_fit.png?raw=true
  :width: 800
  :alt: Alternative text

Simulations
-----------

Simulated interferometer datasets can be generated using the ``SimulatorInterferometer`` object, which includes adding
Gaussian noise to the visibilities:

.. code-block:: python

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=dataset.uv_wavelengths, exposure_time=300.0, noise_sigma=0.01
    )

    real_space_grid = al.Grid2D.uniform(
        shape_native=real_space_mask.shape_native,
        pixel_scales=real_space_mask.pixel_scales,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=real_space_grid)

    dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

Here is the subplot of the simulated interferometer dataset:

.. image:: https://github.com/Jammy2211/PyAutoLens/blob/main/docs/overview/images/overview_6_interferometer/8_subplot_dataset.png?raw=true
  :width: 800
  :alt: Alternative text

Wrap-Up
-------

The `interferometer <https://github.com/Jammy2211/autolens_workspace/tree/release/notebooks/interferometer>`_ package
of the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ contains numerous example scripts for performing
interferometer modeling and simulating strong lens interferometer datasets.