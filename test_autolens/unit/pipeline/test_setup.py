import autolens as al
import autofit as af


def test__lens_light_centre_tag():

    setup = al.SetupPipeline(lens_light_centre=None)
    assert setup.lens_light_centre_tag == ""
    setup = al.SetupPipeline(lens_light_centre=(2.0, 2.0))
    assert setup.lens_light_centre_tag == "__lens_light_centre_(2.00,2.00)"
    setup = al.SetupPipeline(lens_light_centre=(3.0, 4.0))
    assert setup.lens_light_centre_tag == "__lens_light_centre_(3.00,4.00)"
    setup = al.SetupPipeline(lens_light_centre=(3.027, 4.033))
    assert setup.lens_light_centre_tag == "__lens_light_centre_(3.03,4.03)"


def test__lens_mass_centre_tag():

    setup = al.SetupPipeline(lens_mass_centre=None)
    assert setup.lens_mass_centre_tag == ""
    setup = al.SetupPipeline(lens_mass_centre=(2.0, 2.0))
    assert setup.lens_mass_centre_tag == "__lens_mass_centre_(2.00,2.00)"
    setup = al.SetupPipeline(lens_mass_centre=(3.0, 4.0))
    assert setup.lens_mass_centre_tag == "__lens_mass_centre_(3.00,4.00)"
    setup = al.SetupPipeline(lens_mass_centre=(3.027, 4.033))
    assert setup.lens_mass_centre_tag == "__lens_mass_centre_(3.03,4.03)"


def test__align_light_mass_centre_tag__is_empty_sting_if_both_lens_light_and_mass_centres_input():
    setup = al.SetupPipeline(align_light_mass_centre=False)
    assert setup.align_light_mass_centre_tag == ""
    setup = al.SetupPipeline(align_light_mass_centre=True)
    assert setup.align_light_mass_centre_tag == "__align_light_mass_centre"
    setup = al.SetupPipeline(
        lens_light_centre=(0.0, 0.0),
        lens_mass_centre=(1.0, 1.0),
        align_light_mass_centre=True,
    )
    assert setup.align_light_mass_centre_tag == ""


def test__no_shear_tag():
    setup = al.SetupPipeline(no_shear=False)
    assert setup.no_shear_tag == "__with_shear"

    setup = al.SetupPipeline(no_shear=True)
    assert setup.no_shear_tag == "__no_shear"


def test__constant_mass_to_light_ratio_tag():

    setup = al.SetupPipeline(constant_mass_to_light_ratio=True)
    assert setup.constant_mass_to_light_ratio_tag == "_const"
    setup = al.SetupPipeline(constant_mass_to_light_ratio=False)
    assert setup.constant_mass_to_light_ratio_tag == "_free"


def test__bulge_and_disk_mass_to_light_ratio_gradient_tag():

    setup = al.SetupPipeline(bulge_mass_to_light_ratio_gradient=True)
    assert setup.bulge_mass_to_light_ratio_gradient_tag == "_bulge"
    setup = al.SetupPipeline(bulge_mass_to_light_ratio_gradient=False)
    assert setup.bulge_mass_to_light_ratio_gradient_tag == ""

    setup = al.SetupPipeline(disk_mass_to_light_ratio_gradient=True)
    assert setup.disk_mass_to_light_ratio_gradient_tag == "_disk"
    setup = al.SetupPipeline(disk_mass_to_light_ratio_gradient=False)
    assert setup.disk_mass_to_light_ratio_gradient_tag == ""


def test__mass_to_light_ratio_tag():

    setup = al.SetupPipeline(
        constant_mass_to_light_ratio=True,
        bulge_mass_to_light_ratio_gradient=False,
        disk_mass_to_light_ratio_gradient=False,
    )
    assert setup.mass_to_light_ratio_tag == "__mlr_const"

    setup = al.SetupPipeline(
        constant_mass_to_light_ratio=True,
        bulge_mass_to_light_ratio_gradient=True,
        disk_mass_to_light_ratio_gradient=False,
    )
    assert setup.mass_to_light_ratio_tag == "__mlr_const_grad_bulge"

    setup = al.SetupPipeline(
        constant_mass_to_light_ratio=True,
        bulge_mass_to_light_ratio_gradient=True,
        disk_mass_to_light_ratio_gradient=True,
    )
    assert setup.mass_to_light_ratio_tag == "__mlr_const_grad_bulge_disk"


def test__align_light_dark_tag():

    setup = al.SetupPipeline(align_light_dark_centre=False)
    assert setup.align_light_dark_centre_tag == ""
    setup = al.SetupPipeline(align_light_dark_centre=True)
    assert setup.align_light_dark_centre_tag == "__align_light_dark_centre"


def test__align_bulge_dark_tag():
    setup = al.SetupPipeline(align_bulge_dark_centre=False)
    assert setup.align_bulge_dark_centre_tag == ""
    setup = al.SetupPipeline(align_bulge_dark_centre=True)
    assert setup.align_bulge_dark_centre_tag == "__align_bulge_dark_centre"


def test__smbh_tag():

    setup = al.SetupPipeline(include_smbh=False)
    assert setup.include_smbh_tag == ""

    setup = al.SetupPipeline(include_smbh=True, smbh_centre_fixed=True)
    assert setup.include_smbh_tag == "__smbh_centre_fixed"

    setup = al.SetupPipeline(include_smbh=True, smbh_centre_fixed=False)
    assert setup.include_smbh_tag == "__smbh_centre_free"


def test__subhalo_centre_tag():

    setup = al.SetupPipeline(subhalo_instance=None)
    assert setup.subhalo_centre_tag == ""
    setup = al.SetupPipeline(subhalo_instance=al.mp.SphericalNFW(centre=(2.0, 2.0)))
    assert setup.subhalo_centre_tag == "__sub_centre_(2.00,2.00)"
    setup = al.SetupPipeline(subhalo_instance=al.mp.SphericalNFW(centre=(3.0, 4.0)))
    assert setup.subhalo_centre_tag == "__sub_centre_(3.00,4.00)"
    setup = al.SetupPipeline(subhalo_instance=al.mp.SphericalNFW(centre=(3.027, 4.033)))
    assert setup.subhalo_centre_tag == "__sub_centre_(3.03,4.03)"


def test__subhalo_mass_at_200_tag():

    setup = al.SetupPipeline(subhalo_instance=None)
    assert setup.subhalo_mass_at_200_tag == ""
    setup = al.SetupPipeline(
        subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e8)
    )
    assert setup.subhalo_mass_at_200_tag == "__sub_mass_1.0e+08"
    setup = al.SetupPipeline(
        subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e9)
    )
    assert setup.subhalo_mass_at_200_tag == "__sub_mass_1.0e+09"
    setup = al.SetupPipeline(
        subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e10)
    )
    assert setup.subhalo_mass_at_200_tag == "__sub_mass_1.0e+10"


def test__tag():

    setup = al.SetupPipeline(
        hyper_galaxies=True,
        hyper_background_noise=True,
        hyper_image_sky=True,
        align_bulge_dark_centre=True,
    )

    assert (
        setup.tag
        == "setup__hyper_galaxies_bg_sky_bg_noise__with_shear__align_bulge_dark_centre"
    )

    setup = al.SetupPipeline(
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

    setup = al.SetupPipeline(align_light_mass_centre=True, number_of_gaussians=1)

    assert setup.tag == "setup__align_light_mass_centre__gaussians_x1__with_shear"

    setup = al.SetupPipeline(
        subhalo_instance=al.mp.SphericalNFWMCRLudlow(centre=(1.0, 2.0), mass_at_200=1e8)
    )

    assert setup.tag == "setup__with_shear__sub_centre_(1.00,2.00)__sub_mass_1.0e+08"

    setup = al.SetupPipeline(include_smbh=True, smbh_centre_fixed=True)

    assert setup.tag == "setup__with_shear__smbh_centre_fixed"


def test__bulge_light_and_mass_profile():

    light = al.SetupPipeline(bulge_mass_to_light_ratio_gradient=False)
    assert (
        light.bulge_light_and_mass_profile.effective_radius is al.lmp.EllipticalSersic
    )

    light = al.SetupPipeline(bulge_mass_to_light_ratio_gradient=True)
    assert (
        light.bulge_light_and_mass_profile.effective_radius
        is al.lmp.EllipticalSersicRadialGradient
    )


def test__disk_light_and_mass_profile():

    light = al.SetupPipeline(
        disk_as_sersic=False, disk_mass_to_light_ratio_gradient=False
    )
    assert (
        light.disk_light_and_mass_profile.effective_radius
        is al.lmp.EllipticalExponential
    )

    light = al.SetupPipeline(
        disk_as_sersic=True, disk_mass_to_light_ratio_gradient=False
    )
    assert light.disk_light_and_mass_profile.effective_radius is al.lmp.EllipticalSersic

    light = al.SetupPipeline(
        disk_as_sersic=False, disk_mass_to_light_ratio_gradient=True
    )
    assert (
        light.disk_light_and_mass_profile.effective_radius
        is al.lmp.EllipticalExponentialRadialGradient
    )

    light = al.SetupPipeline(
        disk_as_sersic=True, disk_mass_to_light_ratio_gradient=True
    )
    assert (
        light.disk_light_and_mass_profile.effective_radius
        is al.lmp.EllipticalSersicRadialGradient
    )


def test__set_mass_to_light_ratios_of_light_and_mass_profiles():

    lmp_0 = af.PriorModel(al.lmp.EllipticalSersic)
    lmp_1 = af.PriorModel(al.lmp.EllipticalSersic)
    lmp_2 = af.PriorModel(al.lmp.EllipticalSersic)

    setup = al.SetupPipeline(constant_mass_to_light_ratio=False)

    setup.set_mass_to_light_ratios_of_light_and_mass_profiles(
        light_and_mass_profiles=[lmp_0, lmp_1, lmp_2]
    )

    assert lmp_0.mass_to_light_ratio != lmp_1.mass_to_light_ratio
    assert lmp_0.mass_to_light_ratio != lmp_2.mass_to_light_ratio
    assert lmp_1.mass_to_light_ratio != lmp_2.mass_to_light_ratio

    lmp_0 = af.PriorModel(al.lmp.EllipticalSersic)
    lmp_1 = af.PriorModel(al.lmp.EllipticalSersic)
    lmp_2 = af.PriorModel(al.lmp.EllipticalSersic)

    setup = al.SetupPipeline(constant_mass_to_light_ratio=True)

    setup.set_mass_to_light_ratios_of_light_and_mass_profiles(
        light_and_mass_profiles=[lmp_0, lmp_1, lmp_2]
    )

    assert lmp_0.mass_to_light_ratio == lmp_1.mass_to_light_ratio
    assert lmp_0.mass_to_light_ratio == lmp_2.mass_to_light_ratio
    assert lmp_1.mass_to_light_ratio == lmp_2.mass_to_light_ratio


def test__smbh_from_centre():

    setup = al.SetupPipeline(include_smbh=False, smbh_centre_fixed=True)
    smbh = setup.smbh_from_centre(centre=(0.0, 0.0))
    assert smbh is None

    setup = al.SetupPipeline(include_smbh=True, smbh_centre_fixed=True)
    smbh = setup.smbh_from_centre(centre=(0.0, 0.0))
    assert isinstance(smbh, af.PriorModel)
    assert smbh.centre == (0.0, 0.0)

    setup = al.SetupPipeline(include_smbh=True, smbh_centre_fixed=False)
    smbh = setup.smbh_from_centre(centre=(0.1, 0.2), centre_sigma=0.2)
    assert isinstance(smbh, af.PriorModel)
    assert isinstance(smbh.centre[0], af.GaussianPrior)
    assert smbh.centre[0].mean == 0.1
    assert smbh.centre[0].sigma == 0.2
    assert isinstance(smbh.centre[1], af.GaussianPrior)
    assert smbh.centre[1].mean == 0.2
    assert smbh.centre[1].sigma == 0.2
