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

    The convolution uses each image's convolver (*See ccd.convolution*).

    Parameters
    ----------
    unblurred_images_1d : [ndarray]
        List of the 1D masked images which are blurred.
    blurring_images_1d : [ndarray]
        List of the 1D masked blurring images which are used for blurring.
    convolvers : [ccd.convolution.ConvolverImage]
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
    convolver : hyper.ccd.convolution.ConvolverImage
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

def unmasked_blurred_image_of_datas_from_padded_grid_stacks_psfs_and_unmasked_images(padded_grid_stacks, psfs,
                                                                                     unmasked_images_1d):
    """For a list of padded grid-stacks and psf, compute an unmasked blurred image from an unmasked unblurred image \
    for every grid on the stack (e.g. each image in the data-set).

    This relies on using the lens data's padded-grids, which are grids of (y,x) coordinates which extend over the \
    entire image as opposed to just the masked region.

    This returns a list, where each list index corresponds to [data_index].

    Parameters
    ----------
    padded_grid_stacks : [grids.GridStack]
        The list of padded-grid_stacks, whose padded grid are used for PSF convolution.
    psfs : [ccd.PSF]
        The PSF of each image used for convolution.
    unmasked_image_1d : [ndarray]
        The 1D unmasked images which are blurred.
    """

    unmasked_blurred_image_of_datas = []

    for image_index in range(len(padded_grid_stacks)):

        unmasked_blurred_image_of_data = \
            lens_fit_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
            padded_grid_stack=padded_grid_stacks[image_index], psf=psfs[image_index],
            unmasked_image_1d=unmasked_images_1d[image_index])

        unmasked_blurred_image_of_datas.append(unmasked_blurred_image_of_data)

    return unmasked_blurred_image_of_datas

def unmasked_blurred_image_of_datas_and_planes_from_padded_grid_stacks_and_psf(planes, padded_grid_stacks, psfs):
    """For each image in the lens data-set, compute the unmasked blurred image of every unmasked unblurred image of \
    each plane. To do this, this function iterates over all planes to extract their unmasked unblurred images.

    If a galaxy in a plane has a pixelization, the unmasked image is returned as None, as as the inversion's model \
    image cannot be mapped to an unmasked version.

    This relies on using the lens data's padded-grids, which are grids of (y,x) coordinates which extend over the \
    entire image as opposed to just the masked region.

    This returns a list, where each list index corresponds to [data_index][plane_index].

    Parameters
    ----------
    planes : [plane.Plane]
        The list of planes the unmasked blurred images are computed using.
    padded_grid_stacks : [grids.GridStack]
        The list of padded-grid_stacks, whose padded grid are used for PSF convolution.
    psfs : [ccd.PSF]
        The PSF of each image used for convolution.
    """
    unmasked_blurred_image_of_datas_and_planes = []

    for image_index in range(len(padded_grid_stacks)):

        unmasked_blurred_image_of_planes = \
            unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes=planes,
                padded_grid_stack=padded_grid_stacks[image_index], psf=psfs[image_index])

        unmasked_blurred_image_of_datas_and_planes.append(unmasked_blurred_image_of_planes)

    return unmasked_blurred_image_of_datas_and_planes

def unmasked_blurred_image_of_planes_from_padded_grid_stack_and_psf(planes, padded_grid_stack, psf):
    """This is a utility function for the function above, which performs the iteration over each plane'and \
    computes the unmasked blurred image of that plane.

    Parameters
    ----------
    planes : [plane.Plane]
        The list of planes the unmasked blurred images are computed using.
    padded_grid_stack : grids.GridStack
        A padded-grid_stack, whose padded grid is used for PSF convolution.
    psf : ccd.PSF
        The PSF of the image used for convolution.
    """
    unmasked_blurred_image_of_planes = []

    for plane_index, plane in enumerate(planes):

        if plane.has_pixelization:

            unmasked_blurred_image_of_plane = None

        else:

            unmasked_blurred_image_of_plane = \
                lens_fit_util.unmasked_blurred_image_from_padded_grid_stack_psf_and_unmasked_image(
                unmasked_image_1d=plane.image_plane_images_1d[plane_index],
                padded_grid_stack=padded_grid_stack, psf=psf)

        unmasked_blurred_image_of_planes.append(unmasked_blurred_image_of_plane)

    return unmasked_blurred_image_of_planes

def unmasked_blurred_image_of_datas_planes_and_galaxies_from_padded_grid_stacks_and_psf(planes, padded_grid_stacks,
                                                                                        psfs):
    """For each image in the lens data-set, compute the unmasked blurred image of every unmasked unblurred image of \
    every galaxy in each plane. To do this, this function iterates over all planes to extract their unmasked unblurred images.

    If a galaxy in a plane has a pixelization, the unmasked image of the galaxy in that plane is returned as None, \
    as an the inversion's model image cannot be mapped to an unmasked version.

    This relies on using the lens data's padded-grids, which are grids of (y,x) coordinates which extend over the \
    entire image as opposed to just the masked region.

    This returns a list, where each list index corresponds to [data_index][plane_index][galaxy_index].

    Parameters
    ----------
    planes : [plane.Plane]
        The list of planes the unmasked blurred images are computed using.
    padded_grid_stacks : [grids.GridStack]
        The list of padded-grid_stacks, whose padded grid are used for PSF convolution.
    psfs : [ccd.PSF]
        The PSF of each image used for convolution.
    """
    unmasked_blurred_image_of_datas_planes_and_galaxies = []

    for image_index in range(len(padded_grid_stacks)):

        unmasked_blurred_image_of_planes_and_galaxies = \
            unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stacks_and_psfs(planes=planes,
                                    padded_grid_stack=padded_grid_stacks[image_index], psf=psfs[image_index])

        unmasked_blurred_image_of_datas_planes_and_galaxies.append(unmasked_blurred_image_of_planes_and_galaxies)

    return unmasked_blurred_image_of_datas_planes_and_galaxies

def unmasked_blurred_image_of_planes_and_galaxies_from_padded_grid_stacks_and_psfs(planes, padded_grid_stack, psf):
    """This is a utility function for the function above, which performs the iteration over each plane and \
    computes the unmasked blurred image of each galaxy in the plane.

    Parameters
    ----------
    planes : [plane.Plane]
        The list of planes the unmasked blurred images are computed using.
    padded_grid_stack : grids.GridStack
        A padded-grid_stack, whose padded grid is used for PSF convolution.
    psf : ccd.PSF
        The PSF of the image used for convolution.
    """
    unmasked_blurred_image_of_planes_and_galaxies = []

    for plane_index, plane in enumerate(planes):

        unmasked_blurred_image_of_galaxies = \
            lens_fit_util.unmasked_blurred_image_of_galaxies_from_psf_and_unmasked_1d_galaxy_images(
                galaxies=plane.galaxies,
                image_plane_image_1d_of_galaxies=plane.image_plane_images_1d_of_galaxies[plane_index],
                padded_grid_stack=padded_grid_stack, psf=psf)

        unmasked_blurred_image_of_planes_and_galaxies.append(unmasked_blurred_image_of_galaxies)

    return unmasked_blurred_image_of_planes_and_galaxies