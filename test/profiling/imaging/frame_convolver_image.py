import time

import pytest
from profiling import profiling_data

from autolens.model.profiles import light_profiles

subgrid_size = 2

kernel_shape = (39, 39)
sersic = light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.1,
                                         effective_radius=0.8, sersic_index=4.0)

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, subgrid_size=subgrid_size, psf_shape=kernel_shape)
lsst_image = sersic.intensities_from_grid(grid=lsst.coords.image_coords)
lsst_blurring_image = sersic.intensities_from_grid(grid=lsst.coords.blurring_coords)
lsst_kernel_convolver = lsst.masked_image.convolver.convolver_for_kernel(lsst.image_plane_images_.psf.resized_scaled_array_from_array(kernel_shape))

assert lsst_kernel_convolver.convolve_image(image_array=lsst_image, blurring_array=lsst_blurring_image) == \
       pytest.approx(lsst_kernel_convolver.convolve_image(image_array=lsst_image, blurring_array=lsst_blurring_image),
                     1e-4)

euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, subgrid_size=subgrid_size, psf_shape=kernel_shape)
euclid_image = sersic.intensities_from_grid(grid=euclid.coords.image_coords)
euclid_blurring_image = sersic.intensities_from_grid(grid=euclid.coords.blurring_coords)
euclid_kernel_convolver = euclid.masked_image.convolver.convolver_for_kernel(
    euclid.image_plane_images_.psf.resized_scaled_array_from_array(kernel_shape))

assert euclid_kernel_convolver.convolve_image(image_array=euclid_image, blurring_array=euclid_blurring_image) == \
       pytest.approx(
           euclid_kernel_convolver.convolve_image(image_array=euclid_image, blurring_array=euclid_blurring_image), 1e-4)

hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, subgrid_size=subgrid_size, psf_shape=kernel_shape)
hst_image = sersic.intensities_from_grid(grid=hst.coords.image_coords)
hst_blurring_image = sersic.intensities_from_grid(grid=hst.coords.blurring_coords)
hst_kernel_convolver = hst.masked_image.convolver.convolver_for_kernel(hst.image_plane_images_.psf.resized_scaled_array_from_array(kernel_shape))

assert hst_kernel_convolver.convolve_image(image_array=hst_image, blurring_array=hst_blurring_image) == \
       pytest.approx(hst_kernel_convolver.convolve_image(image_array=hst_image, blurring_array=hst_blurring_image),
                     1e-4)

hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, subgrid_size=subgrid_size, psf_shape=kernel_shape)
hst_up_image = sersic.intensities_from_grid(grid=hst_up.coords.image_coords)
hst_up_blurring_image = sersic.intensities_from_grid(grid=hst_up.coords.blurring_coords)
hst_up_kernel_convolver = hst_up.masked_image.convolver.convolver_for_kernel(
    hst_up.image_plane_images_.psf.resized_scaled_array_from_array(kernel_shape))

assert hst_up_kernel_convolver.convolve_image(image_array=hst_up_image, blurring_array=hst_up_blurring_image) == \
       pytest.approx(
           hst_up_kernel_convolver.convolve_image(image_array=hst_up_image, blurring_array=hst_up_blurring_image), 1e-4)

ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, subgrid_size=subgrid_size, psf_shape=kernel_shape)
ao_image = sersic.intensities_from_grid(grid=ao.coords.image_coords)
ao_blurring_image = sersic.intensities_from_grid(grid=ao.coords.blurring_coords)
ao_kernel_convolver = ao.masked_image.convolver.convolver_for_kernel(ao.image_plane_images_.psf.resized_scaled_array_from_array(kernel_shape))

assert ao_kernel_convolver.convolve_image(image_array=ao_image, blurring_array=ao_blurring_image) == \
       pytest.approx(ao_kernel_convolver.convolve_image(image_array=ao_image, blurring_array=ao_blurring_image), 1e-4)

repeats = 1


def tick_toc(func):
    def wrapper():
        start = time.time()
        for _ in range(repeats):
            func()

        diff = time.time() - start
        print("{}: {}".format(func.__name__, diff))

    return wrapper


@tick_toc
def lsst_original_solution():
    lsst_kernel_convolver.convolve_image(image_array=lsst_image, blurring_array=lsst_blurring_image)


@tick_toc
def lsst_jit_solution():
    lsst_kernel_convolver.convolve_image(image_array=lsst_image, blurring_array=lsst_blurring_image)


@tick_toc
def euclid_original_solution():
    euclid_kernel_convolver.convolve_image(image_array=euclid_image, blurring_array=euclid_blurring_image)


@tick_toc
def euclid_jit_solution():
    euclid_kernel_convolver.convolve_image(image_array=euclid_image, blurring_array=euclid_blurring_image)


@tick_toc
def hst_original_solution():
    hst_kernel_convolver.convolve_image(image_array=hst_image, blurring_array=hst_blurring_image)


@tick_toc
def hst_jit_solution():
    hst_kernel_convolver.convolve_image(image_array=hst_image, blurring_array=hst_blurring_image)


@tick_toc
def hst_up_original_solution():
    hst_up_kernel_convolver.convolve_image(image_array=hst_up_image, blurring_array=hst_up_blurring_image)


@tick_toc
def hst_up_jit_solution():
    hst_up_kernel_convolver.convolve_image(image_array=hst_up_image, blurring_array=hst_up_blurring_image)


@tick_toc
def ao_original_solution():
    ao_kernel_convolver.convolve_image(image_array=ao_image, blurring_array=ao_blurring_image)


@tick_toc
def ao_jit_solution():
    ao_kernel_convolver.convolve_image(image_array=ao_image, blurring_array=ao_blurring_image)


# @tick_toc
# def jitted_solution():
#     kernel_convolver.convolve_array_jit(datas)

if __name__ == "__main__":
    lsst_original_solution()
    lsst_jit_solution()

    print()

    euclid_original_solution()
    euclid_jit_solution()

    print()

    hst_original_solution()
    hst_jit_solution()
    #
    print()

    hst_up_original_solution()
    hst_up_jit_solution()

    print()

    ao_original_solution()
    ao_jit_solution()
