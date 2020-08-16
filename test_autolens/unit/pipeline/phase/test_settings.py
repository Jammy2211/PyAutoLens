import autolens as al


class TestTags:
    def test__auto_positions_tag(self):

        settings = al.PhaseSettingsImaging(
            auto_positions_factor=None, auto_positions_minimum_threshold=None
        )
        assert settings.auto_positions_factor_tag == ""
        settings = al.PhaseSettingsImaging(
            auto_positions_factor=1.0, auto_positions_minimum_threshold=None
        )
        assert settings.auto_positions_factor_tag == "__auto_pos_x1.00"
        settings = al.PhaseSettingsImaging(
            auto_positions_factor=2.56, auto_positions_minimum_threshold=None
        )
        assert settings.auto_positions_factor_tag == "__auto_pos_x2.56"
        settings = al.PhaseSettingsImaging(
            auto_positions_factor=None, auto_positions_minimum_threshold=0.5
        )
        assert settings.auto_positions_factor_tag == ""
        settings = al.PhaseSettingsImaging(
            auto_positions_factor=2.56, auto_positions_minimum_threshold=0.5
        )
        assert settings.auto_positions_factor_tag == "__auto_pos_x2.56_min_0.5"

    def test__positions_threshold_tag(self):

        settings = al.PhaseSettingsImaging(positions_threshold=None)
        assert settings.positions_threshold_tag == ""
        settings = al.PhaseSettingsImaging(positions_threshold=1.0)
        assert settings.positions_threshold_tag == "__pos_1.00"
        settings = al.PhaseSettingsImaging(positions_threshold=2.56)
        assert settings.positions_threshold_tag == "__pos_2.56"

    def test__tag__mixture_of_values(self):

        settings = al.PhaseSettingsImaging(
            grid_class=al.Grid,
            grid_inversion_class=al.Grid,
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
            positions_threshold=2.0,
            pixel_scales_interp=None,
        )

        assert (
            settings.phase_no_inversion_tag == "settings__grid_sub_2__snr_2__pos_2.00"
        )
        assert (
            settings.phase_with_inversion_tag
            == "settings__grid_sub_2_inv_sub_2__snr_2__pos_2.00"
        )

        settings = al.PhaseSettingsImaging(
            grid_class=al.GridIterate,
            grid_inversion_class=al.GridInterpolate,
            fractional_accuracy=0.5,
            pixel_scales_interp=0.3,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
            auto_positions_factor=0.5,
            positions_threshold=None,
            pixelization_settings=al.PixelizationSettings(
                use_border=False, is_stochastic=True
            ),
            log_likelihood_cap=200.01,
        )

        assert (
            settings.phase_no_inversion_tag
            == "settings__grid_facc_0.5__bin_3__psf_2x2__auto_pos_x0.50__no_border__stochastic__lh_cap_200.0"
        )
        assert (
            settings.phase_with_inversion_tag
            == "settings__grid_facc_0.5_inv_interp_0.300__bin_3__psf_2x2__auto_pos_x0.50__no_border__stochastic__lh_cap_200.0"
        )

        settings = al.PhaseSettingsInterferometer(
            grid_class=al.GridIterate,
            grid_inversion_class=al.GridInterpolate,
            fractional_accuracy=0.5,
            pixel_scales_interp=0.3,
            transformer_class=al.TransformerDFT,
            primary_beam_shape_2d=(2, 2),
            pixelization_settings=al.PixelizationSettings(
                use_border=False, is_stochastic=True
            ),
            log_likelihood_cap=100.01,
        )

        assert (
            settings.phase_no_inversion_tag
            == "settings__grid_facc_0.5__dft__pb_2x2__no_border__stochastic__lh_cap_100.0"
        )
        assert (
            settings.phase_with_inversion_tag
            == "settings__grid_facc_0.5_inv_interp_0.300__dft__pb_2x2__no_border__stochastic__lh_cap_100.0"
        )

        settings = al.PhaseSettingsInterferometer(
            transformer_class=al.TransformerNUFFT,
            inversion_settings=al.InversionSettings(use_linear_operators=True),
        )

        assert settings.phase_no_inversion_tag == "settings__grid_sub_2__nufft"
        assert (
            settings.phase_with_inversion_tag
            == "settings__grid_sub_2_inv_sub_2__nufft__lop"
        )
