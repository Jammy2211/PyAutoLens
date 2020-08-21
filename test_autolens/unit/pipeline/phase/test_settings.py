import autolens as al


def test__tag__mixture_of_values():

    settings = al.SettingsPhaseImaging(
        settings_masked_imaging=al.SettingsMaskedImaging(
            grid_class=al.Grid,
            grid_inversion_class=al.Grid,
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
        ),
        settings_lens=al.SettingsLens(positions_threshold=2.0),
    )

    assert settings.phase_tag_no_inversion == "settings__grid_sub_2__snr_2__pos_2.00"
    assert (
        settings.phase_tag_with_inversion
        == "settings__grid_sub_2_inv_sub_2__snr_2__pos_2.00"
    )

    settings = al.SettingsPhaseImaging(
        settings_masked_imaging=al.SettingsMaskedImaging(
            grid_class=al.GridIterate,
            grid_inversion_class=al.GridInterpolate,
            fractional_accuracy=0.5,
            pixel_scales_interp=0.3,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
        ),
        settings_lens=al.SettingsLens(auto_positions_factor=0.5),
        settings_pixelization=al.SettingsPixelization(
            use_border=False, is_stochastic=True
        ),
        log_likelihood_cap=200.01,
    )

    assert (
        settings.phase_tag_no_inversion
        == "settings__grid_facc_0.5__bin_3__psf_2x2__auto_pos_x0.50__lh_cap_200.0"
    )
    assert (
        settings.phase_tag_with_inversion
        == "settings__grid_facc_0.5_inv_interp_0.300__bin_3__psf_2x2__auto_pos_x0.50__no_border__stochastic__lh_cap_200.0"
    )

    settings = al.SettingsPhaseInterferometer(
        masked_interferometer=al.SettingsMaskedInterferometer(
            grid_class=al.GridIterate,
            grid_inversion_class=al.GridInterpolate,
            fractional_accuracy=0.5,
            pixel_scales_interp=0.3,
            transformer_class=al.TransformerDFT,
        ),
        settings_pixelization=al.SettingsPixelization(
            use_border=False, is_stochastic=True
        ),
        log_likelihood_cap=100.01,
    )

    assert (
        settings.phase_tag_no_inversion == "settings__grid_facc_0.5__dft__lh_cap_100.0"
    )
    assert (
        settings.phase_tag_with_inversion
        == "settings__grid_facc_0.5_inv_interp_0.300__dft__no_border__stochastic__lh_cap_100.0"
    )

    settings = al.SettingsPhaseInterferometer(
        masked_interferometer=al.SettingsMaskedInterferometer(
            transformer_class=al.TransformerNUFFT
        ),
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

    assert settings.phase_tag_no_inversion == "settings__grid_sub_2__nufft"
    assert (
        settings.phase_tag_with_inversion
        == "settings__grid_sub_2_inv_sub_2__nufft__lop"
    )
