import autolens as al


class TestPipelineGeneralSettings:
    def test__hyper_galaxies_tag(self):

        pipeline_general_settings = al.PipelineGeneralSettings(hyper_galaxies=False)
        assert pipeline_general_settings.hyper_galaxies_tag == ""

        pipeline_general_settings = al.PipelineGeneralSettings(hyper_galaxies=True)
        assert pipeline_general_settings.hyper_galaxies_tag == "_galaxies"

    def test__hyper_image_sky_tag(self):
        pipeline_general_settings = al.PipelineGeneralSettings(hyper_image_sky=False)
        assert pipeline_general_settings.hyper_galaxies_tag == ""

        pipeline_general_settings = al.PipelineGeneralSettings(hyper_image_sky=True)
        assert pipeline_general_settings.hyper_image_sky_tag == "_bg_sky"

    def test__hyper_background_noise_tag(self):
        pipeline_general_settings = al.PipelineGeneralSettings(
            hyper_background_noise=False
        )
        assert pipeline_general_settings.hyper_galaxies_tag == ""

        pipeline_general_settings = al.PipelineGeneralSettings(
            hyper_background_noise=True
        )
        assert pipeline_general_settings.hyper_background_noise_tag == "_bg_noise"

    def test__with_shear_tagger(self):
        pipeline_general_settings = al.PipelineGeneralSettings(with_shear=False)
        assert pipeline_general_settings.hyper_galaxies_tag == ""

        pipeline_general_settings = al.PipelineGeneralSettings(with_shear=True)
        assert pipeline_general_settings.with_shear_tag == "__with_shear"

    def test__tag(self):

        pipeline_general_settings = al.PipelineGeneralSettings(
            hyper_galaxies=True,
            hyper_image_sky=True,
            hyper_background_noise=True,
            with_shear=False,
        )

        assert (
            pipeline_general_settings.tag
            == "pipeline_tag__hyper_galaxies_bg_sky_bg_noise"
        )

        pipeline_general_settings = al.PipelineGeneralSettings(
            hyper_galaxies=True, hyper_background_noise=True, with_shear=True
        )

        assert (
            pipeline_general_settings.tag
            == "pipeline_tag__hyper_galaxies_bg_noise__with_shear"
        )


class TestPipelineSourceSettings:
    def test__pixelization_tagger(self):
        pipeline_source_settings = al.PipelineSourceSettings(pixelization=None)
        assert pipeline_source_settings.pixelization_tag == ""
        pipeline_source_settings = al.PipelineSourceSettings(
            pixelization=al.pix.Rectangular
        )
        assert pipeline_source_settings.pixelization_tag == "__pix_rect"
        pipeline_source_settings = al.PipelineSourceSettings(
            pixelization=al.pix.VoronoiBrightnessImage
        )
        assert pipeline_source_settings.pixelization_tag == "__pix_voro_image"

    def test__regularization_tagger(self):
        pipeline_source_settings = al.PipelineSourceSettings(regularization=None)
        assert pipeline_source_settings.regularization_tag == ""
        pipeline_source_settings = al.PipelineSourceSettings(
            regularization=al.reg.Constant
        )
        assert pipeline_source_settings.regularization_tag == "__reg_const"
        pipeline_source_settings = al.PipelineSourceSettings(
            regularization=al.reg.AdaptiveBrightness
        )
        assert pipeline_source_settings.regularization_tag == "__reg_adapt_bright"

    def test__lens_light_centre_tagger(self):

        pipeline_source_settings = al.PipelineSourceSettings(lens_light_centre=None)
        assert pipeline_source_settings.lens_light_centre_tag == ""
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_light_centre=(2.0, 2.0)
        )
        assert (
            pipeline_source_settings.lens_light_centre_tag
            == "__lens_light_centre_(2.00,2.00)"
        )
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_light_centre=(3.0, 4.0)
        )
        assert (
            pipeline_source_settings.lens_light_centre_tag
            == "__lens_light_centre_(3.00,4.00)"
        )
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_light_centre=(3.027, 4.033)
        )
        assert (
            pipeline_source_settings.lens_light_centre_tag
            == "__lens_light_centre_(3.03,4.03)"
        )

    def test__lens_mass_centre_tagger(self):

        pipeline_source_settings = al.PipelineSourceSettings(lens_mass_centre=None)
        assert pipeline_source_settings.lens_mass_centre_tag == ""
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_mass_centre=(2.0, 2.0)
        )
        assert (
            pipeline_source_settings.lens_mass_centre_tag
            == "__lens_mass_centre_(2.00,2.00)"
        )
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_mass_centre=(3.0, 4.0)
        )
        assert (
            pipeline_source_settings.lens_mass_centre_tag
            == "__lens_mass_centre_(3.00,4.00)"
        )
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_mass_centre=(3.027, 4.033)
        )
        assert (
            pipeline_source_settings.lens_mass_centre_tag
            == "__lens_mass_centre_(3.03,4.03)"
        )

    def test__align_light_mass_centre_tagger__is_empty_sting_if_both_lens_light_and_mass_centres_input(
        self
    ):
        pipeline_source_settings = al.PipelineSourceSettings(
            align_light_mass_centre=False
        )
        assert pipeline_source_settings.align_light_mass_centre_tag == ""
        pipeline_source_settings = al.PipelineSourceSettings(
            align_light_mass_centre=True
        )
        assert (
            pipeline_source_settings.align_light_mass_centre_tag
            == "__align_light_mass_centre"
        )
        pipeline_source_settings = al.PipelineSourceSettings(
            lens_light_centre=(0.0, 0.0),
            lens_mass_centre=(1.0, 1.0),
            align_light_mass_centre=True,
        )
        assert pipeline_source_settings.align_light_mass_centre_tag == ""

    def test__fix_lens_light_tagger(self):
        pipeline_source_settings = al.PipelineSourceSettings(fix_lens_light=False)
        assert pipeline_source_settings.fix_lens_light_tag == ""
        pipeline_source_settings = al.PipelineSourceSettings(fix_lens_light=True)
        assert pipeline_source_settings.fix_lens_light_tag == "__fix_lens_light"

    def test__tag(self):

        pipeline_source_settings = al.PipelineSourceSettings(
            lens_light_centre=(1.0, 2.0),
            lens_mass_centre=(3.0, 4.0),
            align_light_mass_centre=False,
            fix_lens_light=True,
        )

        assert (
            pipeline_source_settings.tag
            == "__pix_voro_image__reg_adapt_bright__lens_light_centre_(1.00,2.00)__lens_mass_centre_(3.00,4.00)__fix_lens_light"
        )
        assert (
            pipeline_source_settings.tag_no_inversion
            == "__lens_light_centre_(1.00,2.00)__lens_mass_centre_(3.00,4.00)__fix_lens_light"
        )

        pipeline_source_settings = al.PipelineSourceSettings(
            align_light_mass_centre=True,
            fix_lens_light=True,
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
        )

        assert (
            pipeline_source_settings.tag
            == "__pix_rect__reg_const__align_light_mass_centre__fix_lens_light"
        )
        assert (
            pipeline_source_settings.tag_no_inversion
            == "__align_light_mass_centre__fix_lens_light"
        )


class TestPipelineLightSettings:
    def test__align_bulge_disk_taggers(self):

        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_centre=False
        )
        assert pipeline_light_settings.align_bulge_disk_centre_tag == ""
        pipeline_light_settings = al.PipelineLightSettings(align_bulge_disk_centre=True)
        assert pipeline_light_settings.align_bulge_disk_centre_tag == "_centre"

        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_axis_ratio=False
        )
        assert pipeline_light_settings.align_bulge_disk_axis_ratio_tag == ""
        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_axis_ratio=True
        )
        assert pipeline_light_settings.align_bulge_disk_axis_ratio_tag == "_axis_ratio"

        pipeline_light_settings = al.PipelineLightSettings(align_bulge_disk_phi=False)
        assert pipeline_light_settings.align_bulge_disk_phi_tag == ""
        pipeline_light_settings = al.PipelineLightSettings(align_bulge_disk_phi=True)
        assert pipeline_light_settings.align_bulge_disk_phi_tag == "_phi"

    def test__bulge_disk_tag(self):
        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_centre=False,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert pipeline_light_settings.align_bulge_disk_tag == ""

        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert (
            pipeline_light_settings.align_bulge_disk_tag == "__align_bulge_disk_centre"
        )

        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=True,
        )
        assert (
            pipeline_light_settings.align_bulge_disk_tag
            == "__align_bulge_disk_centre_phi"
        )

        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            align_bulge_disk_phi=True,
        )
        assert (
            pipeline_light_settings.align_bulge_disk_tag
            == "__align_bulge_disk_centre_axis_ratio_phi"
        )

    def test__disk_as_sersic_tagger(self):
        pipeline_light_settings = al.PipelineLightSettings(disk_as_sersic=False)
        assert pipeline_light_settings.disk_as_sersic_tag == ""
        pipeline_light_settings = al.PipelineLightSettings(disk_as_sersic=True)
        assert pipeline_light_settings.disk_as_sersic_tag == "__disk_sersic"

    def test__tag(self):
        pipeline_light_settings = al.PipelineLightSettings(align_bulge_disk_phi=True)

        assert pipeline_light_settings.tag == "__align_bulge_disk_phi"

        pipeline_light_settings = al.PipelineLightSettings(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            disk_as_sersic=True,
        )

        assert (
            pipeline_light_settings.tag
            == "__align_bulge_disk_centre_axis_ratio__disk_sersic"
        )


class TestPipelineMassSettings:
    def test__align_light_dark_tagger(self):

        pipeline_mass_settings = al.PipelineMassSettings(align_light_dark_centre=False)
        assert pipeline_mass_settings.align_light_dark_centre_tag == ""
        pipeline_mass_settings = al.PipelineMassSettings(align_light_dark_centre=True)
        assert (
            pipeline_mass_settings.align_light_dark_centre_tag
            == "__align_light_dark_centre"
        )

    def test__align_bulge_dark_tagger(self):
        pipeline_mass_settings = al.PipelineMassSettings(align_bulge_dark_centre=False)
        assert pipeline_mass_settings.align_bulge_dark_centre_tag == ""
        pipeline_mass_settings = al.PipelineMassSettings(align_bulge_dark_centre=True)
        assert (
            pipeline_mass_settings.align_bulge_dark_centre_tag
            == "__align_bulge_dark_centre"
        )

    def test__fix_lens_light_tagger(self):
        pipeline_mass_settings = al.PipelineMassSettings(fix_lens_light=False)
        assert pipeline_mass_settings.fix_lens_light_tag == ""
        pipeline_mass_settings = al.PipelineMassSettings(fix_lens_light=True)
        assert pipeline_mass_settings.fix_lens_light_tag == "__fix_lens_light"

    def test__tag(self):

        pipeline_mass_settings = al.PipelineMassSettings(
            align_light_dark_centre=True, fix_lens_light=True
        )

        assert pipeline_mass_settings.tag == "__align_light_dark_centre__fix_lens_light"

        pipeline_mass_settings = al.PipelineMassSettings(align_bulge_dark_centre=True)

        assert pipeline_mass_settings.tag == "__align_bulge_dark_centre"


class TestTags:
    def test__source_tag_from_galaxies(self):

        galaxy = al.Galaxy(redshift=0.5)

        source_tag = al.pipeline_settings.source_tag_from_source(source=galaxy)

        assert source_tag == "parametric"

        galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalExponential(),
            mass=al.mp.EllipticalIsothermal(),
        )

        source_tag = al.pipeline_settings.source_tag_from_source(source=galaxy)

        assert source_tag == "parametric"

        galaxy = al.Galaxy(
            redshift=0.5,
            pixelization=al.pix.VoronoiMagnification(),
            regularization=al.reg.Constant(),
        )

        source_tag = al.pipeline_settings.source_tag_from_source(source=galaxy)

        assert source_tag == "inversion"

        galaxy = al.GalaxyModel(redshift=0.5)

        source_tag = al.pipeline_settings.source_tag_from_source(source=galaxy)

        assert source_tag == "parametric"

        galaxy = al.GalaxyModel(
            redshift=0.5,
            light=al.lp.EllipticalExponential,
            mass=al.mp.EllipticalIsothermal,
        )

        source_tag = al.pipeline_settings.source_tag_from_source(source=galaxy)

        assert source_tag == "parametric"

        galaxy = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        )

        source_tag = al.pipeline_settings.source_tag_from_source(source=galaxy)

        assert source_tag == "inversion"

    # def test__source_from_source(self):
    #
    #     galaxy = al.Galaxy(redshift=0.5, light=al.lp.El)
    #
    #     source = al.pipeline_settings.source_from_source(source=galaxy)
    #
    #     assert source.redshift == 0.5
