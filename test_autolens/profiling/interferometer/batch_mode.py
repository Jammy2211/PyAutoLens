import time

import autolens as al
import numpy as np

repeats = 1

total_visibilities = 500000

real_space_shape_2d = (256, 256)
real_space_pixel_scales = 0.05
real_space_sub_size = 1
real_space_radius = 3.0

image_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]

shape_data = 8 * total_visibilities
shape_preloads = total_visibilities * image_pixels * 2

total_shape = shape_data + shape_preloads

print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
print("PreLoad Memory Use (GB) = " + str(shape_preloads * 8e-9))
print("Total Memory Use (GB) = " + str(total_shape * 8e-9))
print()

# Only delete this if the memory use looks... Okay
# stop

visibilities = al.Visibilities.ones(shape_1d=(total_visibilities,))
uv_wavelengths = np.ones(shape=(total_visibilities, 2))
noise_map = al.VisibilitiesNoiseMap.ones(shape_1d=(total_visibilities,))

interferometer = al.Interferometer(
    visibilities=visibilities, noise_map=noise_map, uv_wavelengths=uv_wavelengths
)

print("Real space sub grid size = " + str(real_space_sub_size))
print("Real space circular mask radius = " + str(real_space_radius) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.EllipticalSersic(intensity=1.0))

mask = al.Mask2D.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    sub_size=real_space_sub_size,
    radius=real_space_radius,
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(fill_value=False, shape=visibilities.shape),
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

image = tracer.image_from_grid(grid=masked_interferometer.grid).in_2d_binned[::-1, :]
images = np.repeat(image[:, :, np.newaxis], 100, axis=2)

transformer = al.TransformerNUFFT(
    uv_wavelengths=interferometer.uv_wavelengths,
    grid=masked_interferometer.grid.in_1d_binned.in_radians,
)

start = time.time()
for i in range(repeats):
    for j in range(100):
        transformer.visibilities_from_image(image=image)
diff = time.time() - start
print("Visibilities Time = {}".format(diff / repeats))


transformer = al.TransformerNUFFT(
    uv_wavelengths=interferometer.uv_wavelengths,
    grid=masked_interferometer.grid.in_1d_binned.in_radians,
    batch=100,
)

start = time.time()
for i in range(repeats):
    transformer.visibilities_from_image_batch(image=images)
diff = time.time() - start
print("Visibilities Time = {}".format(diff / repeats))
