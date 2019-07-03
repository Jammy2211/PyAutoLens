import autofit as af

def pipeline_name_from_name_and_settings(
        pipeline_name, fix_lens_light=False,
        pixelization=None, regularization=None,
        align_bulge_disk_centre=False, align_bulge_disk_axis_ratio=False, align_bulge_disk_phi=False):

    pipeline_tag = pipeline_tag_from_pipeline_settings(
        fix_lens_light=fix_lens_light,
        align_bulge_disk_centre=align_bulge_disk_centre,
        pixelization=pixelization,
        regularization=regularization,
        align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio,
        align_bulge_disk_phi=align_bulge_disk_phi)

    return pipeline_name + pipeline_tag

def pipeline_tag_from_pipeline_settings(
        fix_lens_light=False,
        pixelization=None, regularization=None,
        align_bulge_disk_centre=False, align_bulge_disk_axis_ratio=False, align_bulge_disk_phi=False):

    fix_lens_light_tag = fix_lens_light_tag_from_fix_lens_light(
        fix_lens_light=fix_lens_light)

    pixelization_tag = pixelization_tag_from_pixelization(
        pixelization=pixelization)

    regularization_tag = regularization_tag_from_regularization(
        regularization=regularization)

    bulge_disk_tag = bulge_disk_tag_from_align_bulge_disks(
        align_bulge_disk_centre=align_bulge_disk_centre,
        align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio,
        align_bulge_disk_phi=align_bulge_disk_phi)

    return fix_lens_light_tag + pixelization_tag + regularization_tag + bulge_disk_tag

def fix_lens_light_tag_from_fix_lens_light(fix_lens_light):
    """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
     during he analysis, to customize phase names.

    This changes the phase name 'phase_name' as follows:

    fix_lens_light = False -> phase_name
    fix_lens_light = True -> phase_name_fix_lens_light
    """
    if not fix_lens_light:
        return ''
    elif fix_lens_light:
        return '_fix_lens_light'

def pixelization_tag_from_pixelization(pixelization):

    if pixelization is None:
        return ''
    else:
        return '_pix_' + af.conf.instance.label.get('tag', pixelization().__class__.__name__, str)

def regularization_tag_from_regularization(regularization):

    if regularization is None:
        return ''
    else:
        return '_reg_' + af.conf.instance.label.get('tag', regularization().__class__.__name__, str)
    
def align_bulge_disk_centre_tag_from_align_bulge_disk_centre(align_bulge_disk_centre):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changee the phase name 'phase_name' as follows:

    bd_align_centres = False -> phase_name
    bd_align_centres = True -> phase_name_bd_align_centres
    """
    if not align_bulge_disk_centre:
        return ''
    elif align_bulge_disk_centre:
        return '_bd_align_centre'

def align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(align_bulge_disk_axis_ratio):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changes the phase name 'phase_name' as follows:

    bd_align_axis_ratio = False -> phase_name
    bd_align_axis_ratio = True -> phase_name_bd_align_axis_ratio
    """
    if not align_bulge_disk_axis_ratio:
        return ''
    elif align_bulge_disk_axis_ratio:
        return '_bd_align_axis_ratio'

def align_bulge_disk_phi_tag_from_align_bulge_disk_phi(align_bulge_disk_phi):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changes the phase name 'phase_name' as follows:

    bd_align_phi = False -> phase_name
    bd_align_phi = True -> phase_name_bd_align_phi
    """
    if not align_bulge_disk_phi:
        return ''
    elif align_bulge_disk_phi:
        return '_bd_align_phi'

def bulge_disk_tag_from_align_bulge_disks(align_bulge_disk_centre, align_bulge_disk_axis_ratio,
                                          align_bulge_disk_phi):
    """Generate a tag for the alignment of the geometry of the bulge and disk of a bulge-disk system, to customize \
    phase names based on the bulge-disk model. This adds together the bulge_disk tags generated in the 3 functions
    above
    """
    align_bulge_disk_centre_tag = align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
        align_bulge_disk_centre=align_bulge_disk_centre)
    align_bulge_disk_axis_ratio_tag = align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
        align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio)
    align_bulge_disk_phi_tag = align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
        align_bulge_disk_phi=align_bulge_disk_phi)

    return align_bulge_disk_centre_tag + align_bulge_disk_axis_ratio_tag + align_bulge_disk_phi_tag


def phase_tag_from_phase_settings(sub_grid_size, bin_up_factor, image_psf_shape, inversion_psf_shape,
                                  positions_threshold, inner_mask_radii, interp_pixel_scale, cluster_pixel_scale):
    
    sub_grid_size_tag = sub_grid_size_tag_from_sub_grid_size(sub_grid_size=sub_grid_size)
    bin_up_factor_tag = bin_up_factor_tag_from_bin_up_factor(bin_up_factor=bin_up_factor)
    image_psf_shape_tag = image_psf_shape_tag_from_image_psf_shape(image_psf_shape=image_psf_shape)
    inversion_psf_shape_tag = inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=inversion_psf_shape)
    positions_threshold_tag = positions_threshold_tag_from_positions_threshold(positions_threshold=positions_threshold)
    inner_mask_radii_tag = inner_mask_radii_tag_from_inner_circular_mask_radii(inner_mask_radii=inner_mask_radii)
    interp_pixel_scale_tag = interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=interp_pixel_scale)
    cluster_pixel_scale_tag = cluster_pixel_scale_tag_from_cluster_pixel_scale(cluster_pixel_scale=cluster_pixel_scale)

    return sub_grid_size_tag + bin_up_factor_tag + image_psf_shape_tag + inversion_psf_shape_tag + \
           positions_threshold_tag + inner_mask_radii_tag + interp_pixel_scale_tag + cluster_pixel_scale_tag

def positions_threshold_tag_from_positions_threshold(positions_threshold):
    """Generate a positions threshold tag, to customize phase names based on the threshold that positions are required \
    to trace within one another.

    This changes the phase name 'phase_name' as follows:

    positions_threshold = 1 -> phase_name
    positions_threshold = 2 -> phase_name_positions_threshold_2
    positions_threshold = 2 -> phase_name_positions_threshold_2
    """
    if positions_threshold == None:
        return ''
    else:
        return '_pos_{0:.2f}'.format(positions_threshold)

def sub_grid_size_tag_from_sub_grid_size(sub_grid_size):
    """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

    This changes the phase name 'phase_name' as follows:

    sub_grid_size = None -> phase_name
    sub_grid_size = 1 -> phase_name_sub_grid_size_2
    sub_grid_size = 4 -> phase_name_sub_grid_size_4
    """
    return '_sub_' + str(sub_grid_size)

def inner_mask_radii_tag_from_inner_circular_mask_radii(inner_mask_radii):
    """Generate an inner mask radii tag, to customize phase names based on the size of the circular masked area in the \
    centre of an image.

    This changes the phase name 'phase_name' as follows:

    inner_circular_mask_radii = 1 -> phase_name
    inner_circular_mask_radii = 2 -> phase_name_inner_circular_mask_radii_2
    inner_circular_mask_radii = 2 -> phase_name_inner_circular_mask_radii_2
    """
    if inner_mask_radii == None:
        return ''
    else:
        return '_inner_mask_{0:.2f}'.format(inner_mask_radii)

def image_psf_shape_tag_from_image_psf_shape(image_psf_shape):
    """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
    is trimmed to for faster run times.

    This changes the phase name 'phase_name' as follows:

    image_psf_shape = 1 -> phase_name
    image_psf_shape = 2 -> phase_name_image_psf_shape_2
    image_psf_shape = 2 -> phase_name_image_psf_shape_2
    """
    if image_psf_shape is None:
        return ''
    else:
        y = str(image_psf_shape[0])
        x = str(image_psf_shape[1])
        return ('_image_psf_' + y + 'x' + x)

def inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape):
    """Generate an inversion psf shape tag, to customize phase names based on size of the inversion PSF that the \
    original PSF is trimmed to for faster run times.

    This changes the phase name 'phase_name' as follows:

    inversion_psf_shape = 1 -> phase_name
    inversion_psf_shape = 2 -> phase_name_inversion_psf_shape_2
    inversion_psf_shape = 2 -> phase_name_inversion_psf_shape_2
    """
    if inversion_psf_shape is None:
        return ''
    else:
        y = str(inversion_psf_shape[0])
        x = str(inversion_psf_shape[1])
        return ('_inv_psf_' + y + 'x' + x)

def bin_up_factor_tag_from_bin_up_factor(bin_up_factor):
    """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
    run times.

    This changes the phase name 'phase_name' as follows:

    bin_up_factor = 1 -> phase_name
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    """
    if bin_up_factor == 1 or bin_up_factor is None:
        return ''
    else:
        return '_bin_up_' + str(bin_up_factor)

def interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale):
    """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the interpolation \
    grid that deflection angles are computed on before interpolating to the regular and sub grids.

    This changes the phase name 'phase_name' as follows:

    interp_pixel_scale = 1 -> phase_name
    interp_pixel_scale = 2 -> phase_name_interp_pixel_scale_2
    interp_pixel_scale = 2 -> phase_name_interp_pixel_scale_2
    """
    if interp_pixel_scale is None:
        return ''
    else:
        return '_interp_{0:.3f}'.format(interp_pixel_scale)
    
def cluster_pixel_scale_tag_from_cluster_pixel_scale(cluster_pixel_scale):
    """Generate an clusterolation pixel scale tag, to customize phase names based on the resolution of the clusterolation \
    grid that deflection angles are computed on before clusterolating to the regular and sub grids.

    This changes the phase name 'phase_name' as follows:

    cluster_pixel_scale = 1 -> phase_name
    cluster_pixel_scale = 2 -> phase_name_cluster_pixel_scale_2
    cluster_pixel_scale = 2 -> phase_name_cluster_pixel_scale_2
    """
    if cluster_pixel_scale is None:
        return ''
    else:
        return '_cluster_{0:.3f}'.format(cluster_pixel_scale)