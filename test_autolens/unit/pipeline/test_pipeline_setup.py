import autolens as al


class TestPipelineGeneralSettings:
    def test__hyper_galaxies_tag(self):

        general = al.setup.General(hyper_galaxies=False)
        assert general.hyper_galaxies_tag == ""

        general = al.setup.General(hyper_galaxies=True)
        assert general.hyper_galaxies_tag == "_galaxies"

    def test__hyper_image_sky_tag(self):
        general = al.setup.General(hyper_image_sky=False)
        assert general.hyper_galaxies_tag == ""

        general = al.setup.General(hyper_image_sky=True)
        assert general.hyper_image_sky_tag == "_bg_sky"

    def test__hyper_background_noise_tag(self):
        general = al.setup.General(hyper_background_noise=False)
        assert general.hyper_galaxies_tag == ""

        general = al.setup.General(hyper_background_noise=True)
        assert general.hyper_background_noise_tag == "_bg_noise"

    def test__tag(self):

        general = al.setup.General(
            hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
        )

        assert general.tag == "general__hyper_galaxies_bg_sky_bg_noise"

        general = al.setup.General(hyper_galaxies=True, hyper_background_noise=True)

        assert general.tag == "general__hyper_galaxies_bg_noise"


class TestPipelineSourceSettings:
    def test__pixelization_tag(self):
        source = al.setup.Source(pixelization=None)
        assert source.pixelization_tag == ""
        source = al.setup.Source(pixelization=al.pix.Rectangular)
        assert source.pixelization_tag == "__pix_rect"
        source = al.setup.Source(pixelization=al.pix.VoronoiBrightnessImage)
        assert source.pixelization_tag == "__pix_voro_image"

    def test__regularization_tag(self):
        source = al.setup.Source(regularization=None)
        assert source.regularization_tag == ""
        source = al.setup.Source(regularization=al.reg.Constant)
        assert source.regularization_tag == "__reg_const"
        source = al.setup.Source(regularization=al.reg.AdaptiveBrightness)
        assert source.regularization_tag == "__reg_adapt_bright"

    def test__lens_light_centre_tag(self):

        source = al.setup.Source(lens_light_centre=None)
        assert source.lens_light_centre_tag == ""
        source = al.setup.Source(lens_light_centre=(2.0, 2.0))
        assert source.lens_light_centre_tag == "__lens_light_centre_(2.00,2.00)"
        source = al.setup.Source(lens_light_centre=(3.0, 4.0))
        assert source.lens_light_centre_tag == "__lens_light_centre_(3.00,4.00)"
        source = al.setup.Source(lens_light_centre=(3.027, 4.033))
        assert source.lens_light_centre_tag == "__lens_light_centre_(3.03,4.03)"

    def test__lens_mass_centre_tag(self):

        source = al.setup.Source(lens_mass_centre=None)
        assert source.lens_mass_centre_tag == ""
        source = al.setup.Source(lens_mass_centre=(2.0, 2.0))
        assert source.lens_mass_centre_tag == "__lens_mass_centre_(2.00,2.00)"
        source = al.setup.Source(lens_mass_centre=(3.0, 4.0))
        assert source.lens_mass_centre_tag == "__lens_mass_centre_(3.00,4.00)"
        source = al.setup.Source(lens_mass_centre=(3.027, 4.033))
        assert source.lens_mass_centre_tag == "__lens_mass_centre_(3.03,4.03)"

    def test__align_light_mass_centre_tag__is_empty_sting_if_both_lens_light_and_mass_centres_input(
        self
    ):
        source = al.setup.Source(align_light_mass_centre=False)
        assert source.align_light_mass_centre_tag == ""
        source = al.setup.Source(align_light_mass_centre=True)
        assert source.align_light_mass_centre_tag == "__align_light_mass_centre"
        source = al.setup.Source(
            lens_light_centre=(0.0, 0.0),
            lens_mass_centre=(1.0, 1.0),
            align_light_mass_centre=True,
        )
        assert source.align_light_mass_centre_tag == ""

    def test__lens_light_bulge_only_tag(self):
        source = al.setup.Source(lens_light_bulge_only=False)
        assert source.lens_light_bulge_only_tag == ""
        source = al.setup.Source(lens_light_bulge_only=True)
        assert source.lens_light_bulge_only_tag == "__bulge_only"

    def test__no_shear_tag(self):
        source = al.setup.Source(no_shear=False)
        assert source.no_shear_tag == "__with_shear"

        source = al.setup.Source(no_shear=True)
        assert source.no_shear_tag == "__no_shear"

    def test__fix_lens_light_tag(self):
        source = al.setup.Source(fix_lens_light=False)
        assert source.fix_lens_light_tag == ""
        source = al.setup.Source(fix_lens_light=True)
        assert source.fix_lens_light_tag == "__fix_lens_light"

    def test__tag_from_source(self):

        source = al.setup.Source(
            pixelization=al.pix.VoronoiMagnification, regularization=al.reg.Constant
        )

        galaxy = al.Galaxy(redshift=0.5)

        source_tag = source.tag_from_source(source=galaxy)

        assert source_tag == "source__parametric__with_shear"

        galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalExponential(),
            mass=al.mp.EllipticalIsothermal(),
        )

        source_tag = source.tag_from_source(source=galaxy)

        assert source_tag == "source__parametric__with_shear"

        galaxy = al.Galaxy(
            redshift=0.5,
            pixelization=al.pix.VoronoiMagnification(),
            regularization=al.reg.Constant(),
        )

        source_tag = source.tag_from_source(source=galaxy)

        assert source_tag == "source__pix_voro_mag__reg_const__with_shear"

        galaxy = al.GalaxyModel(redshift=0.5)

        source_tag = source.tag_from_source(source=galaxy)

        assert source_tag == "source__parametric__with_shear"

        galaxy = al.GalaxyModel(
            redshift=0.5,
            light=al.lp.EllipticalExponential,
            mass=al.mp.EllipticalIsothermal,
        )

        source_tag = source.tag_from_source(source=galaxy)

        assert source_tag == "source__parametric__with_shear"

        galaxy = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        )

        source_tag = source.tag_from_source(source=galaxy)

        assert source_tag == "source__pix_voro_mag__reg_const__with_shear"

    def test__tag(self):

        source = al.setup.Source(
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
            lens_light_centre=(1.0, 2.0),
            lens_mass_centre=(3.0, 4.0),
            align_light_mass_centre=False,
            no_shear=True,
            fix_lens_light=True,
        )

        assert (
            source.tag
            == "source__pix_rect__reg_const__no_shear__lens_light_centre_(1.00,2.00)__lens_mass_centre_(3.00,4.00)__fix_lens_light"
        )
        assert (
            source.tag_no_inversion
            == "source__no_shear__lens_light_centre_(1.00,2.00)__lens_mass_centre_(3.00,4.00)__fix_lens_light"
        )
        assert (
            source.tag_beginner
            == "source__pix_rect__reg_const"
        )
        assert (
            source.tag_beginner_no_inversion
            == "source"
        )

        source = al.setup.Source(
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
            align_light_mass_centre=True,
            fix_lens_light=True,
            lens_light_bulge_only=True,
        )

        assert (
            source.tag
            == "source__pix_rect__reg_const__with_shear__align_light_mass_centre__bulge_only__fix_lens_light"
        )
        assert (
            source.tag_no_inversion
            == "source__with_shear__align_light_mass_centre__bulge_only__fix_lens_light"
        )


class TestPipelineLightSettings:
    def test__align_bulge_disk_tags(self):

        light = al.setup.Light(align_bulge_disk_centre=False)
        assert light.align_bulge_disk_centre_tag == ""
        light = al.setup.Light(align_bulge_disk_centre=True)
        assert light.align_bulge_disk_centre_tag == "_centre"

        light = al.setup.Light(align_bulge_disk_axis_ratio=False)
        assert light.align_bulge_disk_axis_ratio_tag == ""
        light = al.setup.Light(align_bulge_disk_axis_ratio=True)
        assert light.align_bulge_disk_axis_ratio_tag == "_axis_ratio"

        light = al.setup.Light(align_bulge_disk_phi=False)
        assert light.align_bulge_disk_phi_tag == ""
        light = al.setup.Light(align_bulge_disk_phi=True)
        assert light.align_bulge_disk_phi_tag == "_phi"

    def test__bulge_disk_tag(self):
        light = al.setup.Light(
            align_bulge_disk_centre=False,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert light.align_bulge_disk_tag == ""

        light = al.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre"

        light = al.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=True,
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre_phi"

        light = al.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            align_bulge_disk_phi=True,
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre_axis_ratio_phi"

    def test__disk_as_sersic_tag(self):
        light = al.setup.Light(disk_as_sersic=False)
        assert light.disk_as_sersic_tag == "__disk_exp"
        light = al.setup.Light(disk_as_sersic=True)
        assert light.disk_as_sersic_tag == "__disk_sersic"

    def test__tag(self):
        light = al.setup.Light(align_bulge_disk_phi=True)

        assert light.tag == "light__align_bulge_disk_phi__disk_exp"

        light = al.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            disk_as_sersic=True,
        )

        assert light.tag == "light__align_bulge_disk_centre_axis_ratio__disk_sersic"


class TestPipelineMassSettings:
    def test__no_shear_tag(self):
        mass = al.setup.Mass(no_shear=False)
        assert mass.no_shear_tag == "__with_shear"

        mass = al.setup.Mass(no_shear=True)
        assert mass.no_shear_tag == "__no_shear"

    def test__align_light_dark_tag(self):

        mass = al.setup.Mass(align_light_dark_centre=False)
        assert mass.align_light_dark_centre_tag == ""
        mass = al.setup.Mass(align_light_dark_centre=True)
        assert mass.align_light_dark_centre_tag == "__align_light_dark_centre"

    def test__align_bulge_dark_tag(self):
        mass = al.setup.Mass(align_bulge_dark_centre=False)
        assert mass.align_bulge_dark_centre_tag == ""
        mass = al.setup.Mass(align_bulge_dark_centre=True)
        assert mass.align_bulge_dark_centre_tag == "__align_bulge_dark_centre"

    def test__fix_lens_light_tag(self):
        mass = al.setup.Mass(fix_lens_light=False)
        assert mass.fix_lens_light_tag == ""
        mass = al.setup.Mass(fix_lens_light=True)
        assert mass.fix_lens_light_tag == "__fix_lens_light"

    def test__tag(self):

        mass = al.setup.Mass(
            no_shear=True, align_light_dark_centre=True, fix_lens_light=True
        )

        assert mass.tag == "mass__no_shear__align_light_dark_centre__fix_lens_light"

        mass = al.setup.Mass(align_bulge_dark_centre=True)

        assert mass.tag == "mass__with_shear__align_bulge_dark_centre"


# class TestTags:

# def test__source_from_source(self):
#
#     galaxy = al.Galaxy(redshift=0.5, light=al.lp.El)
#
#     source = al.pipeline_settings.source_from_source(source=galaxy)
#
#     assert source.redshift == 0.5
