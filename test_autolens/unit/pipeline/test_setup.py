import autolens as al


class TestSetupMass:
    def test__no_shear_tag(self):
        setup = al.SetupMassLightDark(no_shear=False)
        assert setup.no_shear_tag == "__with_shear"

        setup = al.SetupMassLightDark(no_shear=True)
        assert setup.no_shear_tag == "__no_shear"


class TestSetupSubhalo:
    def test__mass_is_model_tag(self):

        setup = al.SetupSubhalo(mass_is_model=False)
        assert setup.mass_is_model_tag == "__mass_is_instance"

        setup = al.SetupSubhalo(mass_is_model=True)
        assert setup.mass_is_model_tag == "__mass_is_model"

    def test__source_is_model_tag(self):

        setup = al.SetupSubhalo(source_is_model=False)
        assert setup.source_is_model_tag == "__source_is_instance"

        setup = al.SetupSubhalo(source_is_model=True)
        assert setup.source_is_model_tag == "__source_is_model"

    def test__grid_size_tag(self):

        setup = al.SetupSubhalo(grid_size=3)
        assert setup.grid_size_tag == "__grid_3"

        setup = al.SetupSubhalo(grid_size=4)
        assert setup.grid_size_tag == "__grid_4"

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

    def test__tag(self):

        setup = al.SetupSubhalo(mass_is_model=True, source_is_model=False)
        assert setup.tag == "subhalo[nfw__mass_is_model__source_is_instance__grid_5]"

        setup = al.SetupSubhalo(
            mass_is_model=False,
            source_is_model=True,
            grid_size=4,
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(
                centre=(2.0, 2.0), mass_at_200=1e10
            ),
        )
        assert (
            setup.tag
            == "subhalo[nfw__mass_is_instance__source_is_model__grid_4__sub_centre_(2.00,2.00)__sub_mass_1.0e+10]"
        )


class TestSetupPipeline:
    def test__tag(self):

        hyper = al.SetupHyper(
            hyper_galaxies=True, hyper_background_noise=True, hyper_image_sky=True
        )

        setup_mass = al.SetupMassLightDark(align_bulge_dark_centre=True)

        setup = al.SetupPipeline(setup_hyper=hyper, setup_mass=setup_mass)

        assert (
            setup.tag == "setup__"
            "hyper[galaxies_bg_sky_bg_noise]__"
            "mass[light_dark__with_shear__mlr_free__align_bulge_dark_centre]"
        )

        setup_source = al.SetupSourceInversion(
            pixelization=al.pix.Rectangular, regularization=al.reg.Constant
        )

        setup_light = al.SetupLightBulgeDisk(light_centre=(1.0, 2.0))

        setup_mass = al.SetupMassLightDark(mass_centre=(3.0, 4.0), no_shear=True)

        setup = al.SetupPipeline(
            setup_source=setup_source, setup_light=setup_light, setup_mass=setup_mass
        )

        assert (
            setup.tag == "setup__"
            "light[bulge_disk__light_centre_(1.00,2.00)]__"
            "mass[light_dark__mass_centre_(3.00,4.00)__no_shear__mlr_free]__"
            "source[pix_rect__reg_const]"
        )

        setup_mass = al.SetupMassLightDark(align_light_dark_centre=True)

        setup = al.SetupPipeline(setup_mass=setup_mass)

        assert (
            setup.tag == "setup__"
            "mass[light_dark__with_shear__mlr_free__align_light_dark_centre]"
        )

        smbh = al.SetupSMBH(include_smbh=True, smbh_centre_fixed=True)

        subhalo = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(
                centre=(1.0, 2.0), mass_at_200=1e8
            )
        )

        setup = al.SetupPipeline(setup_smbh=smbh, subhalo=subhalo)

        assert (
            setup.tag == "setup__"
            "smbh[centre_fixed]__"
            "subhalo[nfw__sub_centre_(1.00,2.00)__sub_mass_1.0e+08]"
        )
