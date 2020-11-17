import autolens as al
import autolens.plot as aplt
import numpy as np

from test_autolens.simulators.interferometer import instrument_util

interferometer = instrument_util.load_test_interferometer(
    dataset_name="mass_sie__source_sersic", instrument="sma"
)

# aplt.Interferometer.visibilities(interferometer=interferometer)
# aplt.Interferometer.uv_wavelengths(interferometer=interferometer)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

# source_galaxy = al.Galaxy(
#     redshift=1.0,
#     light=al.lp.EllipticalSersic(
#         centre=(0.0, 0.0),
#         elliptical_comps=(-0.055555, 0.096225),
#         intensity=0.4,
#         effective_radius=0.5,
#         sersic_index=1.0,
#     ),
# )

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

real_space_shape = 256
real_space_shape_2d = (real_space_shape, real_space_shape)
real_space_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]
real_space_pixel_scales = 0.05

mask = al.Mask2D.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    sub_size=1,
    radius=3.0,
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    transformer_class=al.TransformerNUFFT,
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

inversion = tracer.inversion_interferometer_from_grid_and_data(
    grid=masked_interferometer.grid,
    visibilities=masked_interferometer.visibilities,
    noise_map=masked_interferometer.noise_map,
    transformer=masked_interferometer.transformer,
    settings_inversion=al.SettingsInversion(use_linear_operators=True),
)

aplt.Inversion.reconstruction(inversion=inversion)
