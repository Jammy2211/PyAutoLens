import autolens as al


class TestPhaseTag:
    def test__mixture_of_values(self):

        phase_tag = al.tagging.phase_tag_from_phase_settings(
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
            positions_threshold=2.0,
            pixel_scale_interpolation_grid=None,
        )

        assert phase_tag == "phase_tag__sub_2__snr_2__pos_2.00"

        phase_tag = al.tagging.phase_tag_from_phase_settings(
            sub_size=1,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
            auto_positions_factor=0.5,
            positions_threshold=None,
            pixel_scale_interpolation_grid=0.2,
        )

        assert (
            phase_tag
            == "phase_tag__sub_1__bin_3__psf_2x2__auto_pos_x0.50__interp_0.200"
        )

        phase_tag = al.tagging.phase_tag_from_phase_settings(
            sub_size=1,
            transformer_class=al.TransformerDFT,
            real_space_shape_2d=(3, 3),
            real_space_pixel_scales=(1.0, 2.0),
            primary_beam_shape_2d=(2, 2),
        )

        assert (
            phase_tag == "phase_tag__dft__rs_shape_3x3__rs_pix_1.00x2.00__sub_1__pb_2x2"
        )


class TestPhaseTaggers:
    def test__auto_positions_factor_tagger(self):

        tag = al.tagging.auto_positions_factor_tag_from_auto_positions_factor(
            auto_positions_factor=None
        )
        assert tag == ""
        tag = al.tagging.auto_positions_factor_tag_from_auto_positions_factor(
            auto_positions_factor=1.0
        )
        assert tag == "__auto_pos_x1.00"
        tag = al.tagging.auto_positions_factor_tag_from_auto_positions_factor(
            auto_positions_factor=2.56
        )
        assert tag == "__auto_pos_x2.56"

    def test__positions_threshold_tagger(self):

        tag = al.tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=None
        )
        assert tag == ""
        tag = al.tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=1.0
        )
        assert tag == "__pos_1.00"
        tag = al.tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=2.56
        )
        assert tag == "__pos_2.56"

    def test__pixel_scale_interpolation_grid_tagger(self):

        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=None
        )
        assert tag == ""
        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=0.5
        )
        assert tag == "__interp_0.500"
        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=0.25
        )
        assert tag == "__interp_0.250"
        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=0.234
        )
        assert tag == "__interp_0.234"
