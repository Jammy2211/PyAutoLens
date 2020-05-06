from autoarray.operators import transformer
from autogalaxy.pipeline import tagging


def phase_tag_from_phase_settings(
    sub_size,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    psf_shape_2d=None,
    primary_beam_shape_2d=None,
    auto_positions_factor=None,
    positions_threshold=None,
    pixel_scale_interpolation_grid=None,
    transformer_class=None,
    real_space_shape_2d=None,
    real_space_pixel_scales=None,
):

    phase_tag = tagging.phase_tag_from_phase_settings(
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        psf_shape_2d=psf_shape_2d,
        transformer_class=transformer_class,
        primary_beam_shape_2d=primary_beam_shape_2d,
        real_space_shape_2d=real_space_shape_2d,
        real_space_pixel_scales=real_space_pixel_scales,
    )

    auto_positions_factor_tag = auto_positions_factor_tag_from_auto_positions_factor(
        auto_positions_factor=auto_positions_factor
    )
    positions_threshold_tag = positions_threshold_tag_from_positions_threshold(
        positions_threshold=positions_threshold
    )
    pixel_scale_interpolation_grid_tag = pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
    )

    return (
        phase_tag
        + auto_positions_factor_tag
        + positions_threshold_tag
        + pixel_scale_interpolation_grid_tag
    )


def auto_positions_factor_tag_from_auto_positions_factor(auto_positions_factor):
    """Generate an auto positions factor tag, to customize phase names based on the factor automated positions are
    required to trace within one another.

    This changes the phase name 'phase_name' as follows:

    auto_positions_factor = None -> phase_name
    auto_positions_factor = 2.0 -> phase_name__auto_pos_x2.00
    auto_positions_factor = 3.0 -> phase_name__auto_pos_x3.00
    """
    if auto_positions_factor is None:
        return ""
    return "__auto_pos_x{0:.2f}".format(auto_positions_factor)


def positions_threshold_tag_from_positions_threshold(positions_threshold):
    """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
    to trace within one another.

    This changes the phase name 'phase_name' as follows:

    positions_threshold = 1 -> phase_name
    positions_threshold = 2 -> phase_name_positions_threshold_2
    positions_threshold = 2 -> phase_name_positions_threshold_2
    """
    if positions_threshold is None:
        return ""
    return "__pos_{0:.2f}".format(positions_threshold)


def pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
    pixel_scale_interpolation_grid
):
    """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the interpolation \
    grid that deflection angles are computed on before interpolating to the and sub aa.

    This changes the phase name 'phase_name' as follows:

    pixel_scale_interpolation_grid = 1 -> phase_name
    pixel_scale_interpolation_grid = 2 -> phase_name_pixel_scale_interpolation_grid_2
    pixel_scale_interpolation_grid = 2 -> phase_name_pixel_scale_interpolation_grid_2
    """
    if pixel_scale_interpolation_grid is None:
        return ""
    return "__interp_{0:.3f}".format(pixel_scale_interpolation_grid)
