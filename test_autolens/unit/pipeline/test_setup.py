import autofit as af
import autolens as al


class TestSetupHyper:
    def test__hyper_galaxies_names_and_tag_for_lens_and_source(self):
        setup = al.SetupHyper(hyper_galaxies_lens=False, hyper_galaxies_source=False)
        assert setup.hyper_galaxies is False
        assert setup.hyper_galaxies_tag == ""
        assert setup.hyper_galaxy_names == None

        setup = al.SetupHyper(hyper_galaxies_lens=True, hyper_galaxies_source=False)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxies_tag == "galaxies_lens"
        assert setup.hyper_galaxy_names == ["lens"]

        setup = al.SetupHyper(hyper_galaxies_lens=False, hyper_galaxies_source=True)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxies_tag == "galaxies_source"
        assert setup.hyper_galaxy_names == ["source"]

        setup = al.SetupHyper(hyper_galaxies_lens=True, hyper_galaxies_source=True)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxies_tag == "galaxies_lens_source"
        assert setup.hyper_galaxy_names == ["lens", "source"]

    def test__hyper_fixed_after_source(self):

        hyper = al.SetupHyper(hyper_fixed_after_source=False)
        assert hyper.hyper_fixed_after_source_tag == ""

        hyper = al.SetupHyper(hyper_fixed_after_source=True)
        assert hyper.hyper_fixed_after_source_tag == "__fixed_from_source"

    def test__tag(self):

        setup_hyper = al.SetupHyper(hyper_image_sky=False, hyper_background_noise=False)

        assert setup_hyper.tag == ""

        setup_hyper = al.SetupHyper(hyper_image_sky=True)

        assert setup_hyper.tag == "hyper[__bg_sky]"

        setup_hyper = al.SetupHyper(
            hyper_galaxies_lens=True,
            hyper_galaxies_source=False,
            hyper_image_sky=True,
            hyper_background_noise=True,
        )

        assert setup_hyper.tag == "hyper[galaxies_lens__bg_sky__bg_noise]"

        setup_hyper = al.SetupHyper(
            hyper_galaxies_lens=True,
            hyper_galaxies_source=True,
            hyper_background_noise=True,
            hyper_fixed_after_source=True,
        )

        assert (
            setup_hyper.tag
            == "hyper[galaxies_lens_source__bg_noise__fixed_from_source]"
        )


class TestSetupMass:
    def test__with_shear_tag(self):

        setup_mass = al.SetupMassLightDark(with_shear=True)
        assert setup_mass.with_shear_tag == "__with_shear"

        setup_mass = al.SetupMassLightDark(with_shear=False)
        assert setup_mass.with_shear_tag == "__no_shear"


class TestSetupSource:
    def test__tag(self):

        setup = al.SetupSourceParametric()

        assert setup.tag == "source[parametric__bulge_sersic]"


class TestSetupSubhalo:
    def test__subhalo_prior_model_and_tags(self):

        setup = al.SetupSubhalo()

        assert setup.subhalo_prior_model.cls is al.mp.SphericalNFWMCRLudlow
        assert setup.subhalo_prior_model_tag == "nfw_sph_ludlow"

        setup = al.SetupSubhalo(subhalo_prior_model=af.PriorModel(al.mp.EllipticalNFW))

        assert setup.subhalo_prior_model.cls is al.mp.EllipticalNFW
        assert setup.subhalo_prior_model_tag == "nfw"

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
        assert setup.subhalo_centre_tag == "__centre_(2.00,2.00)"
        setup = al.SetupSubhalo(subhalo_instance=al.mp.SphericalNFW(centre=(3.0, 4.0)))
        assert setup.subhalo_centre_tag == "__centre_(3.00,4.00)"
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFW(centre=(3.027, 4.033))
        )
        assert setup.subhalo_centre_tag == "__centre_(3.03,4.03)"

    def test__subhalo_mass_at_200_tag(self):

        setup = al.SetupSubhalo(subhalo_instance=None)
        assert setup.subhalo_mass_at_200_tag == ""
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e8)
        )
        assert setup.subhalo_mass_at_200_tag == "__mass_1.0e+08"
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e9)
        )
        assert setup.subhalo_mass_at_200_tag == "__mass_1.0e+09"
        setup = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(mass_at_200=1e10)
        )
        assert setup.subhalo_mass_at_200_tag == "__mass_1.0e+10"

    def test__tag(self):

        setup = al.SetupSubhalo(mass_is_model=True, source_is_model=False)
        assert (
            setup.tag
            == "subhalo[nfw_sph_ludlow__mass_is_model__source_is_instance__grid_5]"
        )

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
            == "subhalo[nfw_sph_ludlow__mass_is_instance__source_is_model__grid_4__centre_(2.00,2.00)__mass_1.0e+10]"
        )


class TestSetupPipeline:
    def test__tag(self):

        hyper = al.SetupHyper(
            hyper_galaxies_lens=True, hyper_background_noise=True, hyper_image_sky=True
        )

        setup_mass = al.SetupMassLightDark(align_bulge_dark_centre=True)

        setup = al.SetupPipeline(setup_hyper=hyper, setup_mass=setup_mass)

        assert (
            setup.tag == "setup__"
            "hyper[galaxies_lens__bg_sky__bg_noise]__"
            "mass[light_dark__bulge_sersic__disk_exp__mlr_free__dark_nfw_sph_ludlow__with_shear__align_bulge_dark_centre]"
        )

        setup_source = al.SetupSourceInversion(
            pixelization_prior_model=al.pix.Rectangular,
            regularization_prior_model=al.reg.Constant,
        )

        setup_light = al.SetupLightParametric(light_centre=(1.0, 2.0))

        setup_mass = al.SetupMassLightDark(mass_centre=(3.0, 4.0), with_shear=False)

        setup = al.SetupPipeline(
            setup_source=setup_source, setup_light=setup_light, setup_mass=setup_mass
        )

        assert (
            setup.tag == "setup__"
            "light[parametric__bulge_sersic__disk_exp__align_bulge_disk_centre__centre_(1.00,2.00)]__"
            "mass[light_dark__bulge_sersic__disk_exp__mlr_free__dark_nfw_sph_ludlow__no_shear__centre_(3.00,4.00)]__"
            "source[inversion__pix_rect__reg_const]"
        )

        setup_mass = al.SetupMassLightDark(align_bulge_dark_centre=True)

        setup = al.SetupPipeline(setup_mass=setup_mass)

        assert (
            setup.tag == "setup__"
            "mass[light_dark__bulge_sersic__disk_exp__mlr_free__dark_nfw_sph_ludlow__with_shear__align_bulge_dark_centre]"
        )

        smbh = al.SetupSMBH(smbh_centre_fixed=True)

        setup_subhalo = al.SetupSubhalo(
            subhalo_instance=al.mp.SphericalNFWMCRLudlow(
                centre=(1.0, 2.0), mass_at_200=1e8
            )
        )

        setup = al.SetupPipeline(setup_smbh=smbh, setup_subhalo=setup_subhalo)

        assert (
            setup.tag == "setup__"
            "smbh[point_mass__centre_fixed]__"
            "subhalo[nfw_sph_ludlow__mass_is_model__source_is_model__grid_5__centre_(1.00,2.00)__mass_1.0e+08]"
        )
