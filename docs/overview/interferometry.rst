.. _interferometry:

Interferometry
--------------

Alongside CCD imaging data, **PyAutoLens** supports the modeling of interferometer ``data`` from submillimeter and radio
observatories. The dataset is fitted directly in the uv-plane, circumventing issues that arise when fitting a 'dirty
image' such as correlated noise. To begin, we load an interferometer dataset from fits files:

.. code-block:: bash

    dataset_path = "/path/to/dataset/folder"

    interferometer = al.Interferometer.from_fits(
        visibilities_path=path.join(dataset_path, "visibilities.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    )

    interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.figures(visibilities=True, uv_wavelengths=True)

Here is what the interferometer visibilities and uv wavelength (which represent the interferometer's baselines) looks
like (these are representative of an ALMA dataset with ~ 1 million visibilities):

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/visibilities.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/uv_wavelengths.png
  :width: 400
  :alt: Alternative text

Although interferometer lens modeling is performed in the uv-plane and therefore Fourier space, we still need to define
a 'real-space mask'. This mask defines the grid on which the image of the lensed source galaxy is computed via a
_Tracer_, which when we fit it to ``data`` data in the uv-plane is mapped to Fourier space.

.. code-block:: bash

    real_space_mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
    )

    aplt.Tracer.image(tracer=tracer, grid=real_space_mask.geometry.masked_grid)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/image.png
  :width: 400
  :alt: Alternative text

To perform uv-plane modeling, **PyAutoLens** next Fourier transforms this image from real-sapce to the uv-plane.
This operation uses a *Transformer* object, of which there are multiple available in **PyAutoLens**. This includes
a direct Fourier transform which performs the exact Fourier transformw without approximation.

.. code-block:: bash

    transformer_class = al.TransformerDFT

However, the direct Fourier transform is inefficient. For ~10 million visibilities, it requires **thousands of seconds**
to perform a single transform. To model a lens, we'll perform tens of thousands of transforms, making this approach
unfeasible for high quality ALMA and radio datasets.

For this reason, **PyAutoLens** supports the non-uniform fast fourier transform algorithm
**PyNUFFT** (https://github.com/jyhmiinlin/pynufft), which is significantly faster, being able too perform a Fourier
transform of ~10 million in less than a second!

.. code-block:: bash

    transformer_class = al.TransformerNUFFT

The perform a fit, we follow the same process we did for imaging, creating a *MaskedInterferometer* object which
behaves analogously to a ``MaskImaging`` object.

.. code-block:: bash

    visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

    masked_interferometer = al.MaskedInterferometer(
        interferometer=interferometer,
        visibilities_mask=visibilities_mask,
        real_space_mask=real_space_mask,
        transformer_class=transformer_class,
        inversion_use_linear_operators=True, # We'll cover what this does below.
    )

The masked interferometer can now be used with a *FitInterferometer* object to fit it to a data-set:

.. code-block:: bash

    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer, tracer=tracer
    )

Here is what the image of the tracer looks like before it is Fourier transformed to the uv-plane:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/image_pre_ft.png
  :width: 400
  :alt: Alternative text

And here is what the Fourier transformed model visibilities look like:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/model_visibilities.png
  :width: 400
  :alt: Alternative text

To show the fit to the real and imaginary visibilities, we plot the residuals and chi-squared values as a function uv-distance:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/residual_map_real.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/residual_map_imag.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/chi_squared_map_real.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/chi_squared_map_imag.png
  :width: 400
  :alt: Alternative text

Interferometer ``data`` can also be modeled using pixelized source's, which again perform the source reconstruction by
directly fitting the visibilities in the uv-plane. The source reconstruction is visualized in real space:

Computing this source recontruction would be extremely inefficient if **PyAutoLens** used a traditional approach to
linear algebra which explicitly stored in memory the values required to solve for the source fluxes. In fact, for an
interferomter dataset of ~10 million visibilities this would require **hundreds of GB of memory**!

**PyAutoLens** uses the library **PyLops** (https://pylops.readthedocs.io/en/latest/) to represent this calculation as
a sequence of memory-light linear operators.

The combination of **PyNUFFT** and **PyLops** makes the analysis of ~10 million visibilities from observatories such as
ALMA and JVLA feasible in **PyAutoLens**. However, the largest datasets may still require a degree of augmentation,
averaging or tapering. Rest assured, we are actively working on new solution that will make the analysis of
**hundreds of millions** of visibilities feasible.

Simulated interferometer datasets can be generated using the ``SimulatorInterferometer`` object, which includes adding
Gaussian noise to the visibilities:

.. code-block:: bash

    grid = al.Grid.uniform(shape_2d=(151, 151), pixel_scales=0.05, sub_size=4)

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        background_sky_level=1.0,
        noise_sigma=0.01,
    )

    interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

