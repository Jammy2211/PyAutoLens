import autolens as al
import autofit as af


class TestSetupLight:
    def test__lens_light_centre_tag(self):

        setup = al.SetupLight(lens_light_centre=None)
        assert setup.lens_light_centre_tag == ""
        setup = al.SetupLight(lens_light_centre=(2.0, 2.0))
        assert setup.lens_light_centre_tag == "__lens_light_centre_(2.00,2.00)"
        setup = al.SetupLight(lens_light_centre=(3.0, 4.0))
        assert setup.lens_light_centre_tag == "__lens_light_centre_(3.00,4.00)"
        setup = al.SetupLight(lens_light_centre=(3.027, 4.033))
        assert setup.lens_light_centre_tag == "__lens_light_centre_(3.03,4.03)"


class TestSetupMass:
    def test__lens_mass_centre_tag(self):

        setup = al.SetupMass(lens_mass_centre=None)
        assert setup.lens_mass_centre_tag == ""
        setup = al.SetupMass(lens_mass_centre=(2.0, 2.0))
        assert setup.lens_mass_centre_tag == "__lens_mass_centre_(2.00,2.00)"
        setup = al.SetupMass(lens_mass_centre=(3.0, 4.0))
        assert setup.lens_mass_centre_tag == "__lens_mass_centre_(3.00,4.00)"
        setup = al.SetupMass(lens_mass_centre=(3.027, 4.033))
        assert setup.lens_mass_centre_tag == "__lens_mass_centre_(3.03,4.03)"

    def test__no_shear_tag(self):
        setup = al.SetupMass(no_shear=False)
        assert setup.no_shear_tag == "__with_shear"

        setup = al.SetupMass(no_shear=True)
        assert setup.no_shear_tag == "__no_shear"


class TestSetupSubhalo:
    def test__subhalo_centre_tag(self):

        setup = al.SetupSubhalo(subhalo_instance=None)
        assert setup.subhalo_centre_tag == ""
        setup = al.SetupSubhalo(subhalo_instance=al.mp.SphericalNFW(centre=(2.0, 2.0)))
        assert setup.subhalo_centre_tag == "__sub_centre_(2.00,2.00)"
        setup = al.SetupSubhalo(subhalo_instance=al.mp.SphericalNFW(centre=(3.0, 4.0)))
        assert setup.subhalo_centre_tag == "__sub_centre_(3.00,4.00)"
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFW(centre=(3.027, 4.033))
        )
        assert setup.subhalo_centre_tag == "__sub_centre_(3.03,4.03)"

    def test__subhalo_mass_at_200_tag(self):

        setup = al.SetupSubhalo(subhalo_instance=None)
        assert setup.subhalo_mass_at_200_tag == ""
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e8)
        )
        assert setup.subhalo_mass_at_200_tag == "__sub_mass_1.0e+08"
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e9)
        )
        assert setup.subhalo_mass_at_200_tag == "__sub_mass_1.0e+09"
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e10)
        )
        assert setup.subhalo_mass_at_200_tag == "__sub_mass_1.0e+10"


class TestSetupPipeline:
    def test__tag(self):

        hyper = al.SetupHyper(
            hyper_galaxies=True, hyper_background_noise=True, hyper_image_sky=True
        )

        mass = al.SetupMass(align_bulge_dark_centre=True)

        setup = al.SetupPipeline(hyper=hyper, mass=mass)

        assert (
            setup.tag
            == "setup__hyper_galaxies_bg_sky_bg_noise__with_shear__align_bulge_dark_centre"
        )

        source = al.SetupSource(
            pixelization=al.pix.Rectangular, regularization=al.reg.Constant
        )

        light = al.SetupLight(lens_light_centre=(1.0, 2.0))

        mass = al.SetupMass(
            lens_mass_centre=(3.0, 4.0), align_light_mass_centre=False, no_shear=True
        )

        setup = al.SetupPipeline(source=source, light=light, mass=mass)

        assert (
            setup.tag
            == "setup__pix_rect__reg_const__lens_light_centre_(1.00,2.00)__no_shear__lens_mass_centre_(3.00,4.00)"
        )

        mass = al.SetupMass(align_light_mass_centre=True)

        setup = al.SetupPipeline(mass=mass)

        assert setup.tag == "setup__with_shear__align_light_mass_centre"

        smbh = al.SetupSMBH(include_smbh=True, smbh_centre_fixed=True)

        subhalo = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(
                centre=(1.0, 2.0), mass_at_200=1e8
            )
        )

        setup = al.SetupPipeline(smbh=smbh, subhalo=subhalo)

        assert (
            setup.tag
            == "setup__with_shear__smbh_centre_fixed__sub_centre_(1.00,2.00)__sub_mass_1.0e+08"
        )
