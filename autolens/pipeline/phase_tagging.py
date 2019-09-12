def phase_tag_from_phase_settings(
    sub_size,
    signal_to_noise_limit,
    bin_up_factor,
    psf_shape,
    positions_threshold,
    inner_mask_radii,
    pixel_scale_interpolation_grid,
    pixel_scale_binned_cluster_grid,
):

    sub_size_tag = sub_size_tag_from_sub_size(sub_size=sub_size)
    signal_to_noise_limit_tag = signal_to_noise_limit_tag_from_signal_to_noise_limit(
        signal_to_noise_limit=signal_to_noise_limit
    )
    bin_up_factor_tag = bin_up_factor_tag_from_bin_up_factor(
        bin_up_factor=bin_up_factor
    )
    psf_shape_tag = psf_shape_tag_from_image_psf_shape(psf_shape=psf_shape)
    positions_threshold_tag = positions_threshold_tag_from_positions_threshold(
        positions_threshold=positions_threshold
    )
    inner_mask_radii_tag = inner_mask_radii_tag_from_inner_circular_mask_radii(
        inner_mask_radii=inner_mask_radii
    )
    pixel_scale_interpolation_grid_tag = pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
    )
    pixel_scale_binned_cluster_grid_tag = pixel_scale_binned_cluster_grid_tag_from_pixel_scale_binned_cluster_grid(
        pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid
    )

    return (
        "phase_tag"
        + sub_size_tag
        + signal_to_noise_limit_tag
        + bin_up_factor_tag
        + psf_shape_tag
        + positions_threshold_tag
        + inner_mask_radii_tag
        + pixel_scale_interpolation_grid_tag
        + pixel_scale_binned_cluster_grid_tag
    )


def positions_threshold_tag_from_positions_threshold(positions_threshold):
    """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
    to trace within one another.

    This changes the phase name 'phase_name' as follows:

    positions_threshold = 1 -> phase_name
    positions_threshold = 2 -> phase_name_positions_threshold_2
    positions_threshold = 2 -> phase_name_positions_threshold_2
    """
    if positions_threshold == None:
        return ""
    else:
        return "__pos_{0:.2f}".format(positions_threshold)


def sub_size_tag_from_sub_size(sub_size):
    """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

    This changes the phase name 'phase_name' as follows:

    sub_size = None -> phase_name
    sub_size = 1 -> phase_name_sub_size_2
    sub_size = 4 -> phase_name_sub_size_4
    """
    return "__sub_" + str(sub_size)


def signal_to_noise_limit_tag_from_signal_to_noise_limit(signal_to_noise_limit):
    """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
    the data being fitted.

    This changes the phase name 'phase_name' as follows:

    signal_to_noise_limit = None -> phase_name
    signal_to_noise_limit = 2 -> phase_name_snr_2
    signal_to_noise_limit = 10 -> phase_name_snr_10
    """
    if signal_to_noise_limit is None:
        return ""
    else:
        return "__snr_" + str(signal_to_noise_limit)


def bin_up_factor_tag_from_bin_up_factor(bin_up_factor):
    """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
    run times.

    This changes the phase name 'phase_name' as follows:

    bin_up_factor = 1 -> phase_name
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    """
    if bin_up_factor == 1 or bin_up_factor is None:
        return ""
    else:
        return "__bin_" + str(bin_up_factor)


def inner_mask_radii_tag_from_inner_circular_mask_radii(inner_mask_radii):
    """Generate an inner mask radii tag, to customize phase names based on the size of the circular masked area in the \
    centre of an image.

    This changes the phase name 'phase_name' as follows:

    inner_circular_mask_radii = 1 -> phase_name
    inner_circular_mask_radii = 2 -> phase_name_inner_circular_mask_radii_2
    inner_circular_mask_radii = 2 -> phase_name_inner_circular_mask_radii_2
    """
    if inner_mask_radii == None:
        return ""
    else:
        return "__inner_mask_{0:.2f}".format(inner_mask_radii)


def psf_shape_tag_from_image_psf_shape(psf_shape):
    """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
    is trimmed to for faster run times.

    This changes the phase name 'phase_name' as follows:

    image_psf_shape = 1 -> phase_name
    image_psf_shape = 2 -> phase_name_image_psf_shape_2
    image_psf_shape = 2 -> phase_name_image_psf_shape_2
    """
    if psf_shape is None:
        return ""
    else:
        y = str(psf_shape[0])
        x = str(psf_shape[1])
        return "__psf_" + y + "x" + x


def pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
    pixel_scale_interpolation_grid
):
    """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the interpolation \
    grid that deflection angles are computed on before interpolating to the and sub grids.

    This changes the phase name 'phase_name' as follows:

    pixel_scale_interpolation_grid = 1 -> phase_name
    pixel_scale_interpolation_grid = 2 -> phase_name_pixel_scale_interpolation_grid_2
    pixel_scale_interpolation_grid = 2 -> phase_name_pixel_scale_interpolation_grid_2
    """
    if pixel_scale_interpolation_grid is None:
        return ""
    else:
        return "__interp_{0:.3f}".format(pixel_scale_interpolation_grid)


def pixel_scale_binned_cluster_grid_tag_from_pixel_scale_binned_cluster_grid(
    pixel_scale_binned_cluster_grid
):
    """Generate an clusterolation pixel scale tag, to customize phase names based on the resolution of the clusterolation \
    grid that deflection angles are computed on before clusterolating to the and sub grids.

    This changes the phase name 'phase_name' as follows:

    pixel_scale_binned_cluster_grid = 1 -> phase_name
    pixel_scale_binned_cluster_grid = 2 -> phase_name_pixel_scale_binned_cluster_grid_2
    pixel_scale_binned_cluster_grid = 2 -> phase_name_pixel_scale_binned_cluster_grid_2
    """
    if pixel_scale_binned_cluster_grid is None:
        return ""
    else:
        return "__cluster_{0:.3f}".format(pixel_scale_binned_cluster_grid)
