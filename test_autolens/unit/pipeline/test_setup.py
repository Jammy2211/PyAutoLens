import autolens as al


def test__lens_light_centre_tag():

    setup = al.PipelineSetup(lens_light_centre=None)
    assert setup.lens_light_centre_tag == ""
    setup = al.PipelineSetup(lens_light_centre=(2.0, 2.0))
    assert setup.lens_light_centre_tag == "__lens_light_centre_(2.00,2.00)"
    setup = al.PipelineSetup(lens_light_centre=(3.0, 4.0))
    assert setup.lens_light_centre_tag == "__lens_light_centre_(3.00,4.00)"
    setup = al.PipelineSetup(lens_light_centre=(3.027, 4.033))
    assert setup.lens_light_centre_tag == "__lens_light_centre_(3.03,4.03)"


def test__lens_mass_centre_tag():

    setup = al.PipelineSetup(lens_mass_centre=None)
    assert setup.lens_mass_centre_tag == ""
    setup = al.PipelineSetup(lens_mass_centre=(2.0, 2.0))
    assert setup.lens_mass_centre_tag == "__lens_mass_centre_(2.00,2.00)"
    setup = al.PipelineSetup(lens_mass_centre=(3.0, 4.0))
    assert setup.lens_mass_centre_tag == "__lens_mass_centre_(3.00,4.00)"
    setup = al.PipelineSetup(lens_mass_centre=(3.027, 4.033))
    assert setup.lens_mass_centre_tag == "__lens_mass_centre_(3.03,4.03)"


def test__align_light_mass_centre_tag__is_empty_sting_if_both_lens_light_and_mass_centres_input():
    setup = al.PipelineSetup(align_light_mass_centre=False)
    assert setup.align_light_mass_centre_tag == ""
    setup = al.PipelineSetup(align_light_mass_centre=True)
    assert setup.align_light_mass_centre_tag == "__align_light_mass_centre"
    setup = al.PipelineSetup(
        lens_light_centre=(0.0, 0.0),
        lens_mass_centre=(1.0, 1.0),
        align_light_mass_centre=True,
    )
    assert setup.align_light_mass_centre_tag == ""


def test__no_shear_tag():
    setup = al.PipelineSetup(no_shear=False)
    assert setup.no_shear_tag == "__with_shear"

    setup = al.PipelineSetup(no_shear=True)
    assert setup.no_shear_tag == "__no_shear"


def test__align_light_dark_tag():

    setup = al.slam.Mass(align_light_dark_centre=False)
    assert setup.align_light_dark_centre_tag == ""
    setup = al.slam.Mass(align_light_dark_centre=True)
    assert setup.align_light_dark_centre_tag == "__align_light_dark_centre"


def test__align_bulge_dark_tag():
    setup = al.slam.Mass(align_bulge_dark_centre=False)
    assert setup.align_bulge_dark_centre_tag == ""
    setup = al.slam.Mass(align_bulge_dark_centre=True)
    assert setup.align_bulge_dark_centre_tag == "__align_bulge_dark_centre"


def test__tag():

    setup = al.PipelineSetup(
        pixelization=al.pix.Rectangular,
        regularization=al.reg.Constant,
        lens_light_centre=(1.0, 2.0),
        lens_mass_centre=(3.0, 4.0),
        align_light_mass_centre=False,
        no_shear=True,
    )

    assert (
        setup.tag
        == "setup__pix_rect__reg_const__lens_light_centre_(1.00,2.00)__no_shear__lens_mass_centre_(3.00,4.00)"
    )

    setup = al.PipelineSetup(align_light_mass_centre=True, number_of_gaussians=1)

    assert setup.tag == "setup__align_light_mass_centre__gaussians_x1__with_shear"

    setup = al.PipelineSetup(
        hyper_galaxies=True,
        hyper_background_noise=True,
        hyper_image_sky=True,
        align_bulge_dark_centre=True,
    )

    assert (
        setup.tag
        == "setup__hyper_galaxies_bg_sky_bg_noise__with_shear__align_bulge_dark_centre"
    )
