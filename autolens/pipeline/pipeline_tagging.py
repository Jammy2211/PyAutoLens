import autofit as af


def pipeline_name_from_name_and_settings(
    pipeline_name,
    include_shear=False,
    fix_lens_light=False,
    pixelization=None,
    regularization=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_axis_ratio=False,
    align_bulge_disk_phi=False,
):

    pipeline_tag = pipeline_tag_from_pipeline_settings(
        include_shear=include_shear,
        fix_lens_light=fix_lens_light,
        align_bulge_disk_centre=align_bulge_disk_centre,
        pixelization=pixelization,
        regularization=regularization,
        align_bulge_disk_axis_ratio=align_bulge_disk_axis_ratio,
        align_bulge_disk_phi=align_bulge_disk_phi,
    )

    return pipeline_name + pipeline_tag


def pipeline_tag_from_pipeline_settings(
    include_shear=False,
    fix_lens_light=False,
    pixelization=None,
    regularization=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_axis_ratio=False,
    align_bulge_disk_phi=False,
):

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

    return (
        include_shear_tag
        + fix_lens_light_tag
        + pixelization_tag
        + regularization_tag
        + bulge_disk_tag
    )


def include_shear_tag_from_include_shear(include_shear):
    """Generate a tag for if an external shear is included in the mass model of the pipeline and / or phase are fixed
    to a previous estimate, or varied during he analysis, to customize phase names.

    This changes the phase name 'phase_name' as follows:

    fix_lens_light = False -> phase_name
    fix_lens_light = True -> phase_name_fix_lens_light
    """
    if not include_shear:
        return ""
    elif include_shear:
        return "_with_shear"


def fix_lens_light_tag_from_fix_lens_light(fix_lens_light):
    """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
     during he analysis, to customize phase names.

    This changes the phase name 'phase_name' as follows:

    fix_lens_light = False -> phase_name
    fix_lens_light = True -> phase_name_fix_lens_light
    """
    if not fix_lens_light:
        return ""
    elif fix_lens_light:
        return "_fix_lens_light"


def pixelization_tag_from_pixelization(pixelization):

    if pixelization is None:
        return ""
    else:
        return "_pix_" + af.conf.instance.label.get(
            "tag", pixelization().__class__.__name__, str
        )


def regularization_tag_from_regularization(regularization):

    if regularization is None:
        return ""
    else:
        return "_reg_" + af.conf.instance.label.get(
            "tag", regularization().__class__.__name__, str
        )


def align_bulge_disk_centre_tag_from_align_bulge_disk_centre(align_bulge_disk_centre):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changee the phase name 'phase_name' as follows:

    bd_align_centres = False -> phase_name
    bd_align_centres = True -> phase_name_bd_align_centres
    """
    if not align_bulge_disk_centre:
        return ""
    elif align_bulge_disk_centre:
        return "_bd_align_centre"


def align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
    align_bulge_disk_axis_ratio
):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changes the phase name 'phase_name' as follows:

    bd_align_axis_ratio = False -> phase_name
    bd_align_axis_ratio = True -> phase_name_bd_align_axis_ratio
    """
    if not align_bulge_disk_axis_ratio:
        return ""
    elif align_bulge_disk_axis_ratio:
        return "_bd_align_axis_ratio"


def align_bulge_disk_phi_tag_from_align_bulge_disk_phi(align_bulge_disk_phi):
    """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
    based on the bulge-disk model. This changes the phase name 'phase_name' as follows:

    bd_align_phi = False -> phase_name
    bd_align_phi = True -> phase_name_bd_align_phi
    """
    if not align_bulge_disk_phi:
        return ""
    elif align_bulge_disk_phi:
        return "_bd_align_phi"


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
