import autofit as af


def pipeline_tag_from_pipeline_settings(
    hyper_galaxies=False,
    hyper_image_sky=False,
    hyper_background_noise=False,
    include_shear=False,
    fix_lens_light=False,
    pixelization=None,
    regularization=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_axis_ratio=False,
    align_bulge_disk_phi=False,
    disk_as_sersic=False,
    align_light_dark_centre=False,
    align_bulge_dark_centre=False,
):

    hyper_tag = hyper_tag_from_hyper_settings(
        hyper_galaxies=hyper_galaxies,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    include_shear_tag = include_shear_tag_from_include_shear(
        include_shear=include_shear
    )

    fix_lens_light_tag = fix_lens_light_tag_from_fix_lens_light(
        fix_lens_light=fix_lens_light
    )

    pixelization_tag = pixelization_tag_from_pixelization(pixelization=pixelization)

    regularization_tag = regularization_tag_from_regularization(
        regularization=regularization
    )

    bulge_disk_tag = bulge_disk_tag_from_align_bulge_disks(
        align_bulge_disk_centre=align_bulge_disk_centre,
        align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio,
        align_bulge_disk_phi=align_bulge_disk_phi,
    )

    disk_as_sersic_tag = disk_as_sersic_tag_from_disk_as_sersic(
        disk_as_sersic=disk_as_sersic
    )

    align_light_dark_centre_tag = align_light_dark_centre_tag_from_align_light_dark_centre(
        align_light_dark_centre=align_light_dark_centre
    )

    align_bulge_dark_centre_tag = align_bulge_dark_centre_tag_from_align_bulge_dark_centre(
        align_bulge_dark_centre=align_bulge_dark_centre
    )

    return (
        "pipeline_tag"
        + hyper_tag
        + include_shear_tag
        + fix_lens_light_tag
        + pixelization_tag
        + regularization_tag
        + bulge_disk_tag
        + disk_as_sersic_tag
        + align_light_dark_centre_tag
        + align_bulge_dark_centre_tag
    )


def hyper_galaxies_tag_from_hyper_galaxies(hyper_galaxies):
    """Generate a tag for if hyper_galaxies-galaxies are used in a hyper_galaxies pipeline to customize phase names.

    This changes the phase name 'pipeline_name__' as follows:

    fix_lens_light = False -> pipeline_name__
    fix_lens_light = True -> pipeline_name___hyper_galaxies
    """
    if not hyper_galaxies:
        return ""
    elif hyper_galaxies:
        return "_galaxies"


def hyper_image_sky_tag_from_hyper_image_sky(hyper_image_sky):
    """Generate a tag for if the sky-background is hyper as a hyper_galaxies-parameter in a hyper_galaxies pipeline to
    customize phase names.

    This changes the phase name 'pipeline_name__' as follows:

    fix_lens_light = False -> pipeline_name__
    fix_lens_light = True -> pipeline_name___hyper_bg_sky
    """
    if not hyper_image_sky:
        return ""
    elif hyper_image_sky:
        return "_bg_sky"


def hyper_background_noise_tag_from_hyper_background_noise(hyper_background_noise):
    """Generate a tag for if the background noise is hyper as a hyper_galaxies-parameter in a hyper_galaxies pipeline to
    customize phase names.

    This changes the phase name 'pipeline_name__' as follows:

    fix_lens_light = False -> pipeline_name__
    fix_lens_light = True -> pipeline_name___hyper_bg_noise
    """
    if not hyper_background_noise:
        return ""
    elif hyper_background_noise:
        return "_bg_noise"


def hyper_tag_from_hyper_settings(
    hyper_galaxies, hyper_image_sky, hyper_background_noise
):

    if not any([hyper_galaxies, hyper_image_sky, hyper_background_noise]):
        return ""

    hyper_galaxies_tag = hyper_galaxies_tag_from_hyper_galaxies(
        hyper_galaxies=hyper_galaxies
    )

    hyper_image_sky_tag = hyper_image_sky_tag_from_hyper_image_sky(
        hyper_image_sky=hyper_image_sky
    )

    hyper_background_noise_tag = hyper_background_noise_tag_from_hyper_background_noise(
        hyper_background_noise=hyper_background_noise
    )

    return (
        "__hyper"
        + hyper_galaxies_tag
        + hyper_image_sky_tag
        + hyper_background_noise_tag
    )


def include_shear_tag_from_include_shear(include_shear):
    """Generate a tag for if an external shear is included in the mass model of the pipeline and / or phase are fixed
    to a previous estimate, or varied during he analysis, to customize phase names.

    This changes the phase name 'pipeline_name__' as follows:

    fix_lens_light = False -> pipeline_name__
    fix_lens_light = True -> pipeline_name___with_shear
    """
    if not include_shear:
        return ""
    elif include_shear:
        return "__with_shear"


def fix_lens_light_tag_from_fix_lens_light(fix_lens_light):
    """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
     during he analysis, to customize phase names.

    This changes the phase name 'pipeline_name__' as follows:

    fix_lens_light = False -> pipeline_name__
    fix_lens_light = True -> pipeline_name___fix_lens_light
    """
    if not fix_lens_light:
        return ""
    elif fix_lens_light:
        return "__fix_lens_light"


def pixelization_tag_from_pixelization(pixelization):

    if pixelization is None:
        return ""
    else:
        return "__pix_" + af.conf.instance.label.get(
            "tag", pixelization().__class__.__name__, str
        )


def regularization_tag_from_regularization(regularization):

    if regularization is None:
        return ""
    else:
        return "__reg_" + af.conf.instance.label.get(
            "tag", regularization().__class__.__name__, str
        )


def align_bulge_disk_centre_tag_from_align_bulge_disk_centre(align_bulge_disk_centre):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changee the phase name 'pipeline_name__' as follows:

    bd_align_centres = False -> pipeline_name__
    bd_align_centres = True -> pipeline_name___bd_align_centres
    """
    if not align_bulge_disk_centre:
        return ""
    elif align_bulge_disk_centre:
        return "__bd_align_centre"


def align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
    align_bulge_disk_axis_ratio
):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changes the phase name 'pipeline_name__' as follows:

    bd_align_axis_ratio = False -> pipeline_name__
    bd_align_axis_ratio = True -> pipeline_name___bd_align_axis_ratio
    """
    if not align_bulge_disk_axis_ratio:
        return ""
    elif align_bulge_disk_axis_ratio:
        return "__bd_align_axis_ratio"


def align_bulge_disk_phi_tag_from_align_bulge_disk_phi(align_bulge_disk_phi):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changes the phase name 'pipeline_name__' as follows:

    bd_align_phi = False -> pipeline_name__
    bd_align_phi = True -> pipeline_name___bd_align_phi
    """
    if not align_bulge_disk_phi:
        return ""
    elif align_bulge_disk_phi:
        return "__bd_align_phi"


def bulge_disk_tag_from_align_bulge_disks(
    align_bulge_disk_centre, align_bulge_disk_axis_ratio, align_bulge_disk_phi
):
    """Generate a tag for the alignment of the geometry of the bulge and disk of a bulge-disk system, to customize \
    phase names based on the bulge-disk model. This adds together the bulge_disk tags generated in the 3 functions
    above
    """
    align_bulge_disk_centre_tag = align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
        align_bulge_disk_centre=align_bulge_disk_centre
    )
    align_bulge_disk_axis_ratio_tag = align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
        align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio
    )
    align_bulge_disk_phi_tag = align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
        align_bulge_disk_phi=align_bulge_disk_phi
    )

    return (
        align_bulge_disk_centre_tag
        + align_bulge_disk_axis_ratio_tag
        + align_bulge_disk_phi_tag
    )


def disk_as_sersic_tag_from_disk_as_sersic(disk_as_sersic):
    """Generate a tag for if the disk component of a bulge-disk light profile fit of the pipeline is modeled as a \ 
    Sersic or the default profile of an Exponential.

    This changes the phase name 'pipeline_name__' as follows:

    disk_as_sersic = False -> pipeline_name__
    disk_as_sersic = True -> pipeline_name___disk_as_sersic
    """
    if not disk_as_sersic:
        return ""
    elif disk_as_sersic:
        return "__disk_sersic"


def align_light_dark_centre_tag_from_align_light_dark_centre(align_light_dark_centre):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changee the phase name 'pipeline_name__' as follows:

    bd_align_centres = False -> pipeline_name__
    bd_align_centres = True -> pipeline_name___bd_align_centres
    """
    if not align_light_dark_centre:
        return ""
    elif align_light_dark_centre:
        return "__light_dark_align_centre"


def align_bulge_dark_centre_tag_from_align_bulge_dark_centre(align_bulge_dark_centre):
    """Generate a tag for if the bulge and dark of a bulge-dark system are aligned or not, to customize phase names \
    based on the bulge-dark model. This changee the phase name 'pipeline_name__' as follows:

    bd_align_centres = False -> pipeline_name__
    bd_align_centres = True -> pipeline_name___bd_align_centres
    """
    if not align_bulge_dark_centre:
        return ""
    elif align_bulge_dark_centre:
        return "__bulge_dark_align_centre"
