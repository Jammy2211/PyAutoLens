import time

import autolens as al
import numpy as np
import os
import psutil

repeats = 1

kspace_shape = 512
total_visibilities = list(np.logspace(np.log10(1e2), np.log10(1e9), 8).astype("int"))

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


# vis = np.ones(shape=(total_visibilities, 2))

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

pixelization = al.pix.VoronoiMagnification(shape=pixelization_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

mask = al.Mask2D.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    sub_size=real_space_sub_size,
    radius=real_space_radius,
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

run_times = []
memory_use = []

start_overall = time.time()

for total_vis in total_visibilities:

    vis = np.random.uniform(low=-5.0, high=5.0, size=(total_vis, 2))

    visibilities = al.Visibilities.manual_1d(visibilities=vis)

    uv_wavelengths = np.ones(shape=(total_vis, 2))
    noise_map = al.VisibilitiesNoiseMap.ones(shape_1d=(total_vis,))

    interferometer = al.Interferometer(
        visibilities=visibilities, noise_map=noise_map, uv_wavelengths=uv_wavelengths
    )

    masked_interferometer = al.MaskedInterferometer(
        interferometer=interferometer,
        real_space_mask=mask,
        visibilities_mask=np.full(fill_value=False, shape=visibilities.shape),
        transformer_class=al.TransformerNUFFT,
    )

    start = time.time()

    for i in range(repeats):
        inversion = tracer.inversion_interferometer_from_grid_and_data(
            grid=masked_interferometer.grid,
            visibilities=masked_interferometer.visibilities,
            noise_map=masked_interferometer.noise_map,
            transformer=masked_interferometer.transformer,
        )

    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1e9
    memory_use.append(memory)

    diff = (time.time() - start) / repeats

    print()
    print(f"Run Time {diff} for {total_vis} visibilities")
    print(f"Memory use {memory_use}GB for {total_vis} visibilities")

    run_times.append(diff)

import pickle

with open(f"total_visibilities.pickle", "wb") as f:
    pickle.dump(total_visibilities, f)

with open(f"run_times.pickle", "wb") as f:
    pickle.dump(run_times, f)

with open(f"memory_use.pickle", "wb") as f:
    pickle.dump(memory_use, f)


# import matplotlib.pyplot as plt
#
# print(run_times)
#
# plt.figure(figsize=(9, 7))
# plt.plot(total_visibilities, run_times)
#
# plt.xticks(size=11)
# plt.xscale('log')
# plt.ylabel("Run Time (seconds)", fontsize=18)
# plt.tight_layout()
# plt.show()

# print(x)
