.. _api:

Interferometry
--------------

Alongside CCD imaging data, **PyAutoLens** supports the modeling of interferometer data from submm and radio
observatories. Here, the dataset is fitted directly in the uv-plane, circumventing issues that arise when fitting a
'dirty image' such as correlated noise. To begin, we load an interferometer dataset from fits files:

.. code-block:: bash

    dataset_path = "/path/to/dataset/folder"

    interferometer = al.Interferometer.from_fits(
        visibilities_path=f"{dataset_path}/visibilities.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        uv_wavelengths_path=f"{dataset_path}/uv_wavelengths.fits",
    )

    aplt.Interferometer.visibilities(interferometer=interferometer)
    aplt.Interferometer.uv_wavelengths(interferometer=interferometer)

Here is what the interferometer visibilities and uv wavelength (which represent the interferometer's baselines) looks
like (these are representative of a Square Mile Array dataset):

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/visibilities.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/overview/images/interferometry/uv_wavelengths.png
  :width: 400
  :alt: Alternative text

To perform uv-plane modeling, **PyAutoLens** generates an image of the strong lens system via a *Tracer* and then
Fourier transforms it to the uv-plane. This operation uses a *Transformer* object, of which there are multiple
available in **PyAutoLens**. This includes one which performs a direct Fourier transform and thus suffers no
approximations and a non-uniform fast fourier transform which is significalty quicker for datasets with many
visibilities.

.. code-block:: bash

    transformer = al.TransformerDFT()
    transformer = al.TransformerNUFFT()

The perform a fit, we need two masks, a 'real-space mask' which defines the grid on which the image of the lensed
source galaxy is computed and a 'visibilities mask' which defining which visibilities are omitted from the chi-squared
evaluation. We can use these masks to create a *MaskedInterferometer* object which behaves analogously to a
*MaskedImaging* object.

.. code-block:: bash

    real_space_mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
    )

    visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

    masked_interferometer = al.MaskedInterferometer(
        interferometer=interferometer,
        visibilities_mask=visibilities_mask,
        real_space_mask=real_space_mask,
        transformer_class=aa.TransformerNUFFT,
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

Interferometer data can also be modeled using pixelized source's, which again perform the source reconstruction by
directly fitting the visibilities in the uv-plane. The source reconstruction itself is visualized in real space:

Simulated interferometer datasets can be generated using the *SimulatorInterferometer* object, which includes adding
Gaussian noise to the visibilities:

.. code-block:: bash

    grid = al.Grid.uniform(shape_2d=(151, 151), pixel_scales=0.05, sub_size=4)

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time_map=al.Array.full(fill_value=100.0, shape_2d=grid.shape_2d),
        background_sky_map=al.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
        noise_sigma=0.01,
    )

    interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

