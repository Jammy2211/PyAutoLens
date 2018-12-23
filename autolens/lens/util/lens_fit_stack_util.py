import numpy as np

from autolens.lens.util import lens_fit_util

def map_arrays_1d_to_scaled_arrays(arrays_1d, map_to_scaled_arrays):
    """Map a list of 1d arrays to their masked 2D scaled arrays, using a list of map to scaled array functions.

    Parameters
    -----------
    arrays_1d : [ndarray]
        The list of 1D arrays which are mapped to unmasked 2D scaled-arrays.
    map_to_scaled_arrays : [func]
        A list of functions which maps the 1D lens hyper to its unmasked 2D scaled-array.
    """
    return list(map(lambda array_1d, map_to_scaled_array :
                    map_to_scaled_array(array_1d=array_1d),
                    arrays_1d, map_to_scaled_arrays))

def blurred_images_1d_of_images_from_1d_unblurred_and_bluring_images(unblurred_images_1d, blurring_images_1d, convolvers):
    """For a list of 1D masked images and 1D blurring images (the regions outside the masks whose light blurs \
    into the masks after PSF convolution), use both to compute the blurred image within the masks via PSF convolution.

    The convolution uses each image's convolver (*See imaging.convolution*).

    Parameters
    ----------
    unblurred_images_1d : [ndarray]
        List of the 1D masked images which are blurred.
    blurring_images_1d : [ndarray]
        List of the 1D masked blurring images which are used for blurring.
    convolvers : [imaging.convolution.ConvolverImage]
        List of the image-convolvers which perform the convolutions in 1D.
    """
    blurred_images = []

    for image_index in range(len(unblurred_images_1d)):

        blurred_profile_image = lens_fit_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_images_1d[image_index],
            blurring_image_1d=blurring_images_1d[image_index],
            convolver=convolvers[image_index])

        blurred_images.append(blurred_profile_image)

    return blurred_images

def blurred_images_of_images_and_planes_from_1d_images_and_convolver(total_planes, image_plane_images_1d_of_planes,
                                                                     image_plane_blurring_images_1d_of_planes, convolvers,
                                                                     map_to_scaled_arrays):
    """For a tracer, extract the image-plane image of every plane and blur it with the PSF.

    If none of the galaxies in a plane have a light profie or pixelization (and thus don't have an image) a *None* \
    is used.

    Parameters
    ----------
    total_planes : int
        The total number of planes that blurred images are computed for.
    image_plane_image_1d_of_planes : [[ndarray]]
        For every image and for every plane, the 1D image-plane image.
    image_plane_blurring_image_1d_of_planes : [ndarray]
        For every image and for every plane, the 1D image-plane blurring image.
    convolver : hyper.imaging.convolution.ConvolverImage
        Class which performs the PSF convolution of a masked image in 1D.
    map_to_scaled_array : func
        A function which maps a masked image from 1D to 2D.
    """

    blurred_images_of_planes = []

    for image_index in range(len(convolvers)):

        blurred_image_of_planes = lens_fit_util.blurred_image_of_planes_from_1d_images_and_convolver(
            total_planes=total_planes,
            image_plane_image_1d_of_planes=image_plane_images_1d_of_planes[image_index],
            image_plane_blurring_image_1d_of_planes=image_plane_blurring_images_1d_of_planes[image_index],
            convolver=convolvers[image_index], map_to_scaled_array=map_to_scaled_arrays[image_index])

        blurred_images_of_planes.append(blurred_image_of_planes)

    return blurred_images_of_planes

def unmasked_blurred_images_from_padded_grid_stacks_psfs_and_unmasked_images(padded_grid_stacks, psfs,
                                                                             unmasked_images_1d):
    """For a fitting image, compute an unmasked blurred image from an unmasked unblurred image. Unmasked
    images are used for plotting the results of a model outside a masked region.

    This relies on using a fitting image's padded_grid, which is grid of coordinates which extends over the entire
    image as opposed to just the masked region.

    Parameters
    ----------
    fitting_image_ : fitting.fitting_data.FittingImage
        A padded_grid_stack, whose padded grid is used for PSF convolution.
    unmasked_images_ : [ndarray]
        The 1D unmasked images which are blurred.
    """

    unmasked_blurred_images = []

    for image_index in range(len(padded_grid_stacks)):

        unmasked_blurred_image = \
            lens_fit_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
            padded_grid_stack=padded_grid_stacks[image_index], psf=psfs[image_index],
            unmasked_image_1d=unmasked_images_1d[image_index])

        unmasked_blurred_images.append(unmasked_blurred_image)

    return unmasked_blurred_images

def unmasked_blurred_images_of_images_planes_and_galaxies_from_padded_grid_stacks_and_psf(planes, padded_grid_stacks,
                                                                                          psfs):

    unmasked_blurred_images_of_images_planes_and_galaxies = []

    for image_index in range(len(padded_grid_stacks)):

        unmasked_blurred_image_of_planes_and_galaxies = \
            unmasked_blurred_images_of_planes_and_galaxies_from_padded_grid_stacks_and_psfs(planes=planes,
                                    padded_grid_stack=padded_grid_stacks[image_index], psf=psfs[image_index])

        unmasked_blurred_images_of_images_planes_and_galaxies.append(unmasked_blurred_image_of_planes_and_galaxies)

    return unmasked_blurred_images_of_images_planes_and_galaxies

def unmasked_blurred_images_of_planes_and_galaxies_from_padded_grid_stacks_and_psfs(planes, padded_grid_stack, psf):

    unmasked_blurred_images_of_planes_and_galaxies = []

    for plane_index, plane in enumerate(planes):

        unmasked_blurred_images_of_galaxies = \
            lens_fit_util.unmasked_blurred_images_of_galaxies_from_psf_and_unmasked_1d_galaxy_images(
                galaxies=plane.galaxies,
                image_plane_image_1d_of_galaxies=plane.image_plane_images_1d_of_galaxies[plane_index],
                padded_grid_stack=padded_grid_stack, psf=psf)

        unmasked_blurred_images_of_planes_and_galaxies.append(unmasked_blurred_images_of_galaxies)

    return unmasked_blurred_images_of_planes_and_galaxies