import autofit as af
import autolens as al


class TestSlam:
    def test__lens_light_tag_for_source_pipeline(self):

        hyper = al.slam.HyperSetup()
        source = al.slam.SourceSetup()
        light = al.slam.LightSetup()
        mass = al.slam.MassSetup()

        slam = al.slam.SLaM(hyper=hyper, source=source, light=light, mass=mass)

        assert slam.lens_light_tag_for_source_pipeline == ""

        slam.set_light_type(light_type="sersic")

        assert slam.lens_light_tag_for_source_pipeline == "__light_sersic"


class TestHyper:
    def test__hyper_fixed_after_source(self):
        hyper = al.slam.HyperSetup(hyper_fixed_after_source=False)
        assert hyper.hyper_fixed_after_source_tag == ""

        hyper = al.slam.HyperSetup(hyper_fixed_after_source=True)
        assert hyper.hyper_fixed_after_source_tag == "_fixed"

    def test__hyper_tag(self):

        hyper = al.slam.HyperSetup(
            hyper_galaxies=True,
            hyper_image_sky=True,
            hyper_background_noise=True,
            hyper_fixed_after_source=True,
        )

        assert hyper.hyper_tag == "__hyper_galaxies_bg_sky_bg_noise_fixed"

        hyper = al.slam.HyperSetup(hyper_galaxies=True, hyper_background_noise=True)

        assert hyper.hyper_tag == "__hyper_galaxies_bg_noise"

        hyper = al.slam.HyperSetup(
            hyper_fixed_after_source=True,
            hyper_galaxies=True,
            hyper_background_noise=True,
        )

        assert hyper.hyper_tag == "__hyper_galaxies_bg_noise_fixed"


class TestSource:
    def test__lens_light_bulge_only_tag(self):
        source = al.slam.SourceSetup(lens_light_bulge_only=False)
        assert source.lens_light_bulge_only_tag == ""
        source = al.slam.SourceSetup(lens_light_bulge_only=True)
        assert source.lens_light_bulge_only_tag == "__bulge_only"

    def test__tag(self):

        source = al.slam.SourceSetup(
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
            lens_light_centre=(1.0, 2.0),
            lens_mass_centre=(3.0, 4.0),
            align_light_mass_centre=False,
            no_shear=True,
        )

        source.type_tag = source.inversion_tag

        assert (
            source.tag
            == "source____pix_rect__reg_const__no_shear__lens_light_centre_(1.00,2.00)__lens_mass_centre_(3.00,4.00)"
        )

        source = al.slam.SourceSetup(
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
            align_light_mass_centre=True,
            number_of_gaussians=1,
            lens_light_bulge_only=True,
        )

        source.type_tag = "test"

        assert (
            source.tag
            == "source__test__gaussians_x1__with_shear__align_light_mass_centre__bulge_only"
        )

    def test__shear(self):

        source = al.slam.SourceSetup(no_shear=False)
        assert source.shear is al.mp.ExternalShear
        source = al.slam.SourceSetup(no_shear=True)
        assert source.shear == None

    def test__align_centre_of_mass_to_light(self):

        mass = af.PriorModel(al.mp.SphericalIsothermal)

        source = al.slam.SourceSetup(align_light_mass_centre=False)

        mass = source.align_centre_of_mass_to_light(mass=mass, light_centre=(1.0, 2.0))

        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1
        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1

        source = al.slam.SourceSetup(align_light_mass_centre=True)

        mass = source.align_centre_of_mass_to_light(mass=mass, light_centre=(1.0, 2.0))

        assert mass.centre == (1.0, 2.0)

    def test__align_centre_to_lens_light_centre(self):

        light = af.PriorModel(al.mp.SphericalIsothermal)

        source = al.slam.SourceSetup(lens_light_centre=(1.0, 2.0))

        light = source.align_centre_to_lens_light_centre(light=light)

        assert light.centre == (1.0, 2.0)

    def test__align_centre_to_lens_mass_centre(self):

        mass = af.PriorModel(al.mp.SphericalIsothermal)

        source = al.slam.SourceSetup(lens_mass_centre=(1.0, 2.0))

        mass = source.align_centre_to_lens_mass_centre(mass=mass)

        assert mass.centre == (1.0, 2.0)

    def test__remove_disk_from_lens_galaxy(self):

        lens = al.GalaxyModel(
            redshift=0.5, bulge=al.lp.EllipticalSersic, disk=al.lp.EllipticalExponential
        )

        source = al.slam.SourceSetup(lens_light_bulge_only=False)

        lens = source.remove_disk_from_lens_galaxy(lens=lens)

        assert type(lens.disk) is af.PriorModel

        source = al.slam.SourceSetup(lens_light_bulge_only=True)

        lens = source.remove_disk_from_lens_galaxy(lens=lens)

        assert lens.disk is None

    def test__is_inversion(self):

        source = al.slam.SourceSetup()

        source.type_tag = "sersic"
        assert source.is_inversion == False

        source.type_tag = "anything_else"
        assert source.is_inversion == True

    def test__unfix_lens_mass_centre(self):

        mass = af.PriorModel(al.mp.SphericalIsothermal)
        mass.centre = (1.0, 2.0)

        source = al.slam.SourceSetup()

        mass = source.unfix_lens_mass_centre(mass=mass)

        assert mass.centre == (1.0, 2.0)

        mass = af.PriorModel(al.mp.SphericalIsothermal)
        source = al.slam.SourceSetup(lens_mass_centre=(5.0, 6.0))

        mass = source.unfix_lens_mass_centre(mass=mass)

        assert mass.centre.centre_0.mean == 5.0
        assert mass.centre.centre_0.sigma == 0.05
        assert mass.centre.centre_1.mean == 6.0
        assert mass.centre.centre_1.sigma == 0.05


class TestLight:
    def test__tag(self):

        light = al.slam.LightSetup(align_bulge_disk_elliptical_comps=True)
        light.type_tag = ""

        assert light.tag == "light____align_bulge_disk_ell"

        light = al.slam.LightSetup(align_bulge_disk_centre=True, disk_as_sersic=True)

        light.type_tag = "lol"

        assert light.tag == "light__lol__align_bulge_disk_centre__disk_sersic"

        light = al.slam.LightSetup(
            align_bulge_disk_centre=True,
            align_bulge_disk_elliptical_comps=True,
            disk_as_sersic=True,
            number_of_gaussians=2,
        )
        light.type_tag = "test"

        assert light.tag == "light__test__gaussians_x2"


class TestMass:
    def test__fix_lens_light_tag(self):
        mass = al.slam.MassSetup(fix_lens_light=False)
        assert mass.fix_lens_light_tag == ""
        mass = al.slam.MassSetup(fix_lens_light=True)
        assert mass.fix_lens_light_tag == "__fix_lens_light"

    def test__tag(self):

        mass = al.slam.MassSetup(
            no_shear=True, align_light_dark_centre=True, fix_lens_light=True
        )
        mass.type_tag = ""

        assert mass.tag == "mass____no_shear__align_light_dark_centre__fix_lens_light"

        mass = al.slam.MassSetup(align_bulge_dark_centre=True)

        mass.type_tag = "test"

        assert mass.tag == "mass__test__with_shear__align_bulge_dark_centre"

    def test__shear_from_previous_pipeline(self):

        mass = al.slam.MassSetup(no_shear=True)

        assert mass.shear_from_previous_pipeline == None

        mass = al.slam.MassSetup(no_shear=False)

        assert isinstance(mass.shear_from_previous_pipeline, af.AbstractPromise)
