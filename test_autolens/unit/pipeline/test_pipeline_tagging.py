import autolens as al


class TestPipelineNameTag:
    def test__pipeline_tag__mixture_of_values(self):
        pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
            hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
        )

        assert pipeline_tag == "pipeline_tag__hyper_galaxies_bg_sky_bg_noise"

        pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
            lens_light_centre=(1.0, 2.0), lens_mass_centre=(3.0, 4.0), fix_lens_light=True
        )

        assert pipeline_tag == "pipeline_tag__lens_light_centre_(1.00,2.00)__lens_mass_centre_(3.00,4.00)__fix_lens_light"

        pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True,
            pixelization=al.pix.Rectangular,
            regularization=al.reg.Constant,
        )

        assert pipeline_tag == "pipeline_tag__fix_lens_light__pix_rect__reg_const"

        pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True, align_bulge_disk_phi=True
        )

        assert pipeline_tag == "pipeline_tag__fix_lens_light__bulge_disk_align_phi"

        pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
            align_light_dark_centre=True,
            align_bulge_dark_centre=True,
            disk_as_sersic=True,
        )

        assert (
            pipeline_tag
            == "pipeline_tag__disk_sersic__light_dark_align_centre__bulge_dark_align_centre"
        )


class TestHyperPipelineTaggers:
    def test__hyper_galaxies_tagger(self):
        tag = al.pipeline_tagging.hyper_galaxies_tag_from_hyper_galaxies(
            hyper_galaxies=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.hyper_galaxies_tag_from_hyper_galaxies(
            hyper_galaxies=True
        )
        assert tag == "_galaxies"

    def test__hyper_image_sky_tagger(self):
        tag = al.pipeline_tagging.hyper_image_sky_tag_from_hyper_image_sky(
            hyper_image_sky=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.hyper_image_sky_tag_from_hyper_image_sky(
            hyper_image_sky=True
        )
        assert tag == "_bg_sky"

    def test__hyper_background_noise_tagger(self):
        tag = al.pipeline_tagging.hyper_background_noise_tag_from_hyper_background_noise(
            hyper_background_noise=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.hyper_background_noise_tag_from_hyper_background_noise(
            hyper_background_noise=True
        )
        assert tag == "_bg_noise"

    def test__tag_from_hyper_settings(self):
        tag = al.pipeline_tagging.hyper_tag_from_hyper_settings(
            hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
        )

        assert tag == ""

        tag = al.pipeline_tagging.hyper_tag_from_hyper_settings(
            hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=False
        )

        assert tag == "__hyper_galaxies"

        tag = al.pipeline_tagging.hyper_tag_from_hyper_settings(
            hyper_galaxies=False, hyper_image_sky=True, hyper_background_noise=True
        )

        assert tag == "__hyper_bg_sky_bg_noise"


class TestPipelineTaggers:
    def test__align_light_mass_centre_tagger(self):
        tag = al.pipeline_tagging.align_light_mass_centre_tag_from_align_light_mass_centre(
            initialize_align_light_mass_centre=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.align_light_mass_centre_tag_from_align_light_mass_centre(
            initialize_align_light_mass_centre=True
        )
        assert tag == "__align_light_mass_centre"

    def test__lens_light_centre_tagger(self):

        tag = al.pipeline_tagging.lens_light_centre_tag_from_lens_light_centre(lens_light_centre=None)
        assert tag == ""
        tag = al.pipeline_tagging.lens_light_centre_tag_from_lens_light_centre(lens_light_centre=(2.0, 2.0))
        assert tag == "__lens_light_centre_(2.00,2.00)"
        tag = al.pipeline_tagging.lens_light_centre_tag_from_lens_light_centre(lens_light_centre=(3.0, 4.0))
        assert tag == "__lens_light_centre_(3.00,4.00)"
        tag = al.pipeline_tagging.lens_light_centre_tag_from_lens_light_centre(lens_light_centre=(3.027, 4.033))
        assert tag == "__lens_light_centre_(3.03,4.03)"

    def test__lens_mass_centre_tagger(self):

        tag = al.pipeline_tagging.lens_mass_centre_tag_from_lens_mass_centre(lens_mass_centre=None)
        assert tag == ""
        tag = al.pipeline_tagging.lens_mass_centre_tag_from_lens_mass_centre(lens_mass_centre=(2.0, 2.0))
        assert tag == "__lens_mass_centre_(2.00,2.00)"
        tag = al.pipeline_tagging.lens_mass_centre_tag_from_lens_mass_centre(lens_mass_centre=(3.0, 4.0))
        assert tag == "__lens_mass_centre_(3.00,4.00)"
        tag = al.pipeline_tagging.lens_mass_centre_tag_from_lens_mass_centre(lens_mass_centre=(3.027, 4.033))
        assert tag == "__lens_mass_centre_(3.03,4.03)"

    def test__with_shear_tagger(self):
        tag = al.pipeline_tagging.with_shear_tag_from_with_shear(
            with_shear=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.with_shear_tag_from_with_shear(
            with_shear=True
        )
        assert tag == "__with_shear"

    def test__fix_lens_light_tagger(self):
        tag = al.pipeline_tagging.fix_lens_light_tag_from_fix_lens_light(
            fix_lens_light=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.fix_lens_light_tag_from_fix_lens_light(
            fix_lens_light=True
        )
        assert tag == "__fix_lens_light"

    def test__pixelization_tagger(self):
        tag = al.pipeline_tagging.pixelization_tag_from_pixelization(pixelization=None)
        assert tag == ""
        tag = al.pipeline_tagging.pixelization_tag_from_pixelization(
            pixelization=al.pix.Rectangular
        )
        assert tag == "__pix_rect"
        tag = al.pipeline_tagging.pixelization_tag_from_pixelization(
            pixelization=al.pix.VoronoiBrightnessImage
        )
        assert tag == "__pix_voro_image"

    def test__regularization_tagger(self):
        tag = al.pipeline_tagging.regularization_tag_from_regularization(
            regularization=None
        )
        assert tag == ""
        tag = al.pipeline_tagging.regularization_tag_from_regularization(
            regularization=al.reg.Constant
        )
        assert tag == "__reg_const"
        tag = al.pipeline_tagging.regularization_tag_from_regularization(
            regularization=al.reg.AdaptiveBrightness
        )
        assert tag == "__reg_adapt_bright"

    def test__align_bulge_disk_taggers(self):
        tag = al.pipeline_tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
            align_bulge_disk_centre=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
            align_bulge_disk_centre=True
        )
        assert tag == "__bulge_disk_align_centre"

        tag = al.pipeline_tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=True
        )
        assert tag == "__bulge_disk_align_axis_ratio"

        tag = al.pipeline_tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
            align_bulge_disk_phi=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
            align_bulge_disk_phi=True
        )
        assert tag == "__bulge_disk_align_phi"

    def test__bulge_disk_tag(self):
        tag = al.pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=False,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert tag == ""

        tag = al.pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert tag == "__bulge_disk_align_centre"

        tag = al.pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=True,
        )
        assert tag == "__bulge_disk_align_centre__bulge_disk_align_phi"

        tag = al.pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            align_bulge_disk_phi=True,
        )
        assert (
            tag
            == "__bulge_disk_align_centre__bulge_disk_align_axis_ratio__bulge_disk_align_phi"
        )

    def test__disk_as_sersic_tagger(self):
        tag = al.pipeline_tagging.disk_as_sersic_tag_from_disk_as_sersic(
            disk_as_sersic=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.disk_as_sersic_tag_from_disk_as_sersic(
            disk_as_sersic=True
        )
        assert tag == "__disk_sersic"

    def test__align_light_dark_tagger(self):
        tag = al.pipeline_tagging.align_light_dark_centre_tag_from_align_light_dark_centre(
            align_light_dark_centre=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.align_light_dark_centre_tag_from_align_light_dark_centre(
            align_light_dark_centre=True
        )
        assert tag == "__light_dark_align_centre"

    def test__align_bulge_dark_tagger(self):
        tag = al.pipeline_tagging.align_bulge_dark_centre_tag_from_align_bulge_dark_centre(
            align_bulge_dark_centre=False
        )
        assert tag == ""
        tag = al.pipeline_tagging.align_bulge_dark_centre_tag_from_align_bulge_dark_centre(
            align_bulge_dark_centre=True
        )
        assert tag == "__bulge_dark_align_centre"
