import time

import autolens as al
import numpy as np

repeats = 3

kspace_shape = 512
total_visibilities = 100000

real_space_shape = 256
real_space_shape_2d = (real_space_shape, real_space_shape)
real_space_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]
real_space_pixel_scales = 0.05
real_space_sub_size = 1
real_space_radius = 3.0
pixelization_shape_2d = (30, 30)

image_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]
source_pixels = pixelization_shape_2d[0] * pixelization_shape_2d[1]

shape_data = 8 * total_visibilities
shape_preloads = total_visibilities * image_pixels * 2
shape_mapping_matrix = total_visibilities * source_pixels

total_shape = shape_data + shape_preloads + shape_mapping_matrix

print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
print("PreLoad Memory Use (GB) = " + str(shape_preloads * 8e-9))
print("Mapping Matrix Memory Use (GB) = " + str(shape_mapping_matrix * 8e-9))
print("Total Memory Use (GB) = " + str(total_shape * 8e-9))
print()

# Only delete this if the memory use looks... Okay
# stop

vis = np.ones(shape=(total_visibilities, 2))
vis = np.random.uniform(low=-5.0, high=5.0, size=(total_visibilities, 2))

visibilities = al.Visibilities.manual_1d(visibilities=vis)

uv_wavelengths = np.ones(shape=(total_visibilities, 2))
noise_map = al.VisibilitiesNoiseMap.ones(shape_1d=(total_visibilities,))

interferometer = al.Interferometer(
    visibilities=visibilities, noise_map=noise_map, uv_wavelengths=uv_wavelengths
)

print("Real space sub grid size = " + str(real_space_sub_size))
print("Real space circular mask radius = " + str(real_space_radius) + "\n")
print("pixelization shape = " + str(pixelization_shape_2d) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, phi=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

pixelization = al.pix.VoronoiMagnification(shape=pixelization_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=10.0),
)

mask = al.Mask2D.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    sub_size=real_space_sub_size,
    radius=real_space_radius,
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    transformer_class=al.TransformerNUFFT,
)

print("Number of points = " + str(masked_interferometer.grid.sub_shape_1d) + "\n")
print(
    "Number of visibilities = "
    + str(masked_interferometer.visibilities.shape_1d)
    + "\n"
)

start_overall = time.time()

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()

for i in range(repeats):
    inversion = tracer.inversion_interferometer_from_grid_and_data(
        grid=masked_interferometer.grid,
        visibilities=masked_interferometer.visibilities,
        noise_map=masked_interferometer.noise_map,
        transformer=masked_interferometer.transformer,
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

diff = time.time() - start
print("Time to compute inversion = {}".format(diff / repeats))

start = time.time()

for i in range(repeats):
    inversion.mapped_reconstructed_visibilities

diff = time.time() - start
print("Time to compute inversion mapped = {}".format(diff / repeats))

start = time.time()

for i in range(repeats):
    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

print(fit.log_evidence)

diff = time.time() - start
print("Time to compute fit = {}".format(diff / repeats))


start = time.time()

for i in range(repeats):
    inversion = tracer.inversion_interferometer_from_grid_and_data(
        grid=masked_interferometer.grid,
        visibilities=masked_interferometer.visibilities,
        noise_map=masked_interferometer.noise_map,
        transformer=masked_interferometer.transformer,
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

diff = time.time() - start
print("Time to compute inversion = {}".format(diff / repeats))
