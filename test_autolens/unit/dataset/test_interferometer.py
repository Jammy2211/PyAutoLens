import autolens as al
import numpy as np
import pytest


class TestMaskedInterferometer:
    def test__masked_dataset_via_autoarray(
        self,
        interferometer_7,
        sub_mask_7x7,
        visibilities_mask_7x2,
        visibilities_7x2,
        noise_map_7x2,
    ):

        masked_interferometer_7 = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
            settings=al.SettingsMaskedInterferometer(
                transformer_class=al.TransformerDFT
            ),
        )

        assert (
            masked_interferometer_7.visibilities == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer_7.visibilities == visibilities_7x2).all()

        assert (masked_interferometer_7.noise_map == noise_map_7x2).all()

        assert (
            masked_interferometer_7.visibilities_mask
            == np.full(fill_value=False, shape=(7, 2))
        ).all()

        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == interferometer_7.uv_wavelengths
        ).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths[0, 0]
            == -55636.4609375
        )

        assert type(masked_interferometer_7.transformer) == al.TransformerDFT

    def test__inheritance_via_autoarray(
        self,
        interferometer_7,
        sub_mask_7x7,
        visibilities_mask_7x2,
        grid_7x7,
        sub_grid_7x7,
    ):

        masked_interferometer_7 = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
            settings=al.SettingsMaskedInterferometer(grid_class=al.Grid),
        )

        assert (masked_interferometer_7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_interferometer_7.grid == sub_grid_7x7).all()

        grid = al.Grid.from_mask(mask=sub_mask_7x7)

        assert (masked_interferometer_7.grid == grid).all()

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self,
    ):
        interferometer = al.Interferometer(
            visibilities=al.Visibilities.ones(shape_1d=(19,)),
            noise_map=al.Visibilities.full(fill_value=2.0, shape_1d=(19,)),
            uv_wavelengths=3.0 * np.ones((19, 2)),
        )

        visibilities_mask = np.full(fill_value=False, shape=(19,))

        real_space_mask = al.Mask2D.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        real_space_mask[9, 9] = False

        masked_interferometer = al.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        assert (masked_interferometer.visibilities.in_1d == np.ones((19, 2))).all()
        assert (masked_interferometer.noise_map.in_1d == 2.0 * np.ones((19, 2))).all()
        assert (
            masked_interferometer.interferometer.uv_wavelengths
            == 3.0 * np.ones((19, 2))
        ).all()

    def test__modified_noise_map(
        self, noise_map_7x2, interferometer_7, sub_mask_7x7, visibilities_mask_7x2
    ):

        masked_interferometer_7 = al.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
        )

        noise_map_7x2[0, 0] = 10.0

        masked_interferometer_7 = masked_interferometer_7.modify_noise_map(
            noise_map=noise_map_7x2
        )

        assert masked_interferometer_7.noise_map[0, 0] == 10.0


class TestSimulatorInterferometer:
    def test__from_tracer__same_as_tracer_input(self):

        grid = al.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(intensity=1.0),
            mass=al.mp.EllipticalIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            background_sky_level=100.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

        interferometer_via_image = simulator.from_image(
            image=tracer.image_from_grid(grid=grid)
        )

        assert (
            interferometer.visibilities == interferometer_via_image.visibilities
        ).all()
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer.noise_map == interferometer_via_image.noise_map).all()

    def test__from_deflections_and_galaxies__same_as_calculation_using_tracer(self):

        grid = al.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.EllipticalIsothermal(einstein_radius=1.6)
        )

        source_galaxy = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            background_sky_level=100.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_deflections_and_galaxies(
            deflections=tracer.deflections_from_grid(grid=grid),
            galaxies=[source_galaxy],
        )

        interferometer_via_image = simulator.from_image(
            image=tracer.image_from_grid(grid=grid)
        )

        assert (
            interferometer.visibilities == interferometer_via_image.visibilities
        ).all()
        assert (
            interferometer_via_image.uv_wavelengths == interferometer.uv_wavelengths
        ).all()
        assert (interferometer.noise_map == interferometer_via_image.noise_map).all()

    def test__simulate_interferometer_from_lens__source_galaxy__compare_to_interferometer(
        self,
    ):

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        source_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = al.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.05, sub_size=1)

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            background_sky_level=100.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_galaxies_and_grid(
            galaxies=[lens_galaxy, source_galaxy], grid=grid
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        interferometer_via_image = simulator.from_image(
            image=tracer.image_from_grid(grid=grid)
        )

        assert interferometer.visibilities == pytest.approx(
            interferometer_via_image.visibilities, 1.0e-4
        )
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer_via_image.noise_map == interferometer.noise_map).all()
