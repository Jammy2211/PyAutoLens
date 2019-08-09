from autolens.pipeline import phase_tagging


class TestPhaseTag:
    def test__mixture_of_values(self):

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_grid_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            image_psf_shape=None,
            inversion_psf_shape=None,
            inner_mask_radii=0.3,
            positions_threshold=2.0,
            interp_pixel_scale=None,
            cluster_pixel_scale=None,
        )

        assert phase_tag == "phase_tag__sub_2__snr_2__pos_2.00__inner_mask_0.30"

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_grid_size=1,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            image_psf_shape=(2, 2),
            inversion_psf_shape=(3, 3),
            inner_mask_radii=None,
            positions_threshold=None,
            interp_pixel_scale=0.2,
            cluster_pixel_scale=0.3,
        )

        assert (
            phase_tag
            == "phase_tag__sub_1__bin_3__image_psf_2x2__inv_psf_3x3__interp_0.200__cluster_0.300"
        )


class TestPhaseTaggers:
    def test__positions_threshold_tagger(self):

        tag = phase_tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=None
        )
        assert tag == ""
        tag = phase_tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=1.0
        )
        assert tag == "__pos_1.00"
        tag = phase_tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=2.56
        )
        assert tag == "__pos_2.56"

    def test__inner_circular_mask_radii_tagger(self):

        tag = phase_tagging.inner_mask_radii_tag_from_inner_circular_mask_radii(
            inner_mask_radii=None
        )
        assert tag == ""
        tag = phase_tagging.inner_mask_radii_tag_from_inner_circular_mask_radii(
            inner_mask_radii=0.2
        )
        print(tag)
        assert tag == "__inner_mask_0.20"
        tag = phase_tagging.inner_mask_radii_tag_from_inner_circular_mask_radii(
            inner_mask_radii=3
        )
        assert tag == "__inner_mask_3.00"

    def test__sub_grid_size_tagger(self):

        tag = phase_tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=1)
        assert tag == "__sub_1"
        tag = phase_tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=2)
        assert tag == "__sub_2"
        tag = phase_tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=4)
        assert tag == "__sub_4"

    def test__signal_to_noise_limit_tagger(self):

        tag = phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=None
        )
        assert tag == ""
        tag = phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=1
        )
        assert tag == "__snr_1"
        tag = phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=2
        )
        assert tag == "__snr_2"
        tag = phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=3
        )
        assert tag == "__snr_3"

    def test__bin_up_factor_tagger(self):

        tag = phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=None)
        assert tag == ""
        tag = phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=1)
        assert tag == ""
        tag = phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=2)
        assert tag == "__bin_2"
        tag = phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=3)
        assert tag == "__bin_3"

    def test__image_psf_shape_tagger(self):

        tag = phase_tagging.image_psf_shape_tag_from_image_psf_shape(
            image_psf_shape=None
        )
        assert tag == ""
        tag = phase_tagging.image_psf_shape_tag_from_image_psf_shape(
            image_psf_shape=(2, 2)
        )
        assert tag == "__image_psf_2x2"
        tag = phase_tagging.image_psf_shape_tag_from_image_psf_shape(
            image_psf_shape=(3, 4)
        )
        assert tag == "__image_psf_3x4"

    def test__inversion_psf_shape_tagger(self):

        tag = phase_tagging.inversion_psf_shape_tag_from_inversion_psf_shape(
            inversion_psf_shape=None
        )
        assert tag == ""
        tag = phase_tagging.inversion_psf_shape_tag_from_inversion_psf_shape(
            inversion_psf_shape=(2, 2)
        )
        assert tag == "__inv_psf_2x2"
        tag = phase_tagging.inversion_psf_shape_tag_from_inversion_psf_shape(
            inversion_psf_shape=(3, 4)
        )
        assert tag == "__inv_psf_3x4"

    def test__interp_pixel_scale_tagger(self):

        tag = phase_tagging.interp_pixel_scale_tag_from_interp_pixel_scale(
            interp_pixel_scale=None
        )
        assert tag == ""
        tag = phase_tagging.interp_pixel_scale_tag_from_interp_pixel_scale(
            interp_pixel_scale=0.5
        )
        assert tag == "__interp_0.500"
        tag = phase_tagging.interp_pixel_scale_tag_from_interp_pixel_scale(
            interp_pixel_scale=0.25
        )
        assert tag == "__interp_0.250"
        tag = phase_tagging.interp_pixel_scale_tag_from_interp_pixel_scale(
            interp_pixel_scale=0.234
        )
        assert tag == "__interp_0.234"

    def test__cluster_pixel_scale_tagger(self):

        tag = phase_tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(
            cluster_pixel_scale=None
        )
        assert tag == ""
        tag = phase_tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(
            cluster_pixel_scale=0.5
        )
        assert tag == "__cluster_0.500"
        tag = phase_tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(
            cluster_pixel_scale=0.25
        )
        assert tag == "__cluster_0.250"
        tag = phase_tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(
            cluster_pixel_scale=0.234
        )
        assert tag == "__cluster_0.234"
