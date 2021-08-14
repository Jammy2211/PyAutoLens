from os import path
import numpy as np

import autofit as af
import autolens as al
import pytest

directory = path.dirname(path.realpath(__file__))


class TestAnalysisAbstract:

    pass


class TestAnalysisDataset:
    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, masked_imaging_7x7
    ):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=100.0)
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    pixelization=al.pix.Rectangular(shape=(3, 3)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            )
        )

        masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
            settings=al.SettingsImaging(sub_size_inversion=2)
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_pixelization=al.SettingsPixelization(use_border=True),
        )

        analysis.dataset.grid_inversion[4] = np.array([[500.0, 0.0]])

        instance = model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.fit_imaging_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.source_grid_slim[4][0] == pytest.approx(
            97.19584, 1.0e-2
        )
        assert fit.inversion.mapper.source_grid_slim[4][1] == pytest.approx(
            -3.699999, 1.0e-2
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_pixelization=al.SettingsPixelization(use_border=False),
        )

        analysis.dataset.grid_inversion[4] = np.array([300.0, 0.0])

        instance = model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.fit_imaging_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.source_grid_slim[4][0] == pytest.approx(
            200.0, 1.0e-4
        )

    def test__analysis_no_positions__removes_positions_and_threshold(
        self, masked_imaging_7x7
    ):

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
            positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]),
            settings_lens=al.SettingsLens(positions_threshold=0.01),
        )

        assert analysis.no_positions.positions == None
        assert analysis.no_positions.settings_lens.positions_threshold == None
