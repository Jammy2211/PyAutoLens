from os import path
import numpy as np

import autofit as af
import autolens as al
import pytest
from autolens.analysis import result as res
from autolens.mock import mock

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


class TestAnalysisPoint:
    def test__make_result__result_imaging_is_returned(self, point_dict):

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
            )
        )

        search = mock.MockSearch(name="test_search")

        solver = mock.MockPositionsSolver(
            model_positions=point_dict["point_0"].positions
        )

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultPoint)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, positions_x2, positions_x2_noise_map
    ):

        point_dataset = al.PointDataset(
            name="point_0",
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset])

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(redshift=0.5, point_0=al.ps.Point(centre=(0.0, 0.0)))
            )
        )

        solver = mock.MockPositionsSolver(model_positions=positions_x2)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        assert fit_positions.chi_squared == 0.0
        assert fit_positions.log_likelihood == analysis_log_likelihood

        model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
        solver = mock.MockPositionsSolver(model_positions=model_positions)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        assert fit_positions.residual_map.in_list == [1.0, 1.0]
        assert fit_positions.chi_squared == 2.0
        assert fit_positions.log_likelihood == analysis_log_likelihood

    def test__figure_of_merit__includes_fit_fluxes(
        self, positions_x2, positions_x2_noise_map, fluxes_x2, fluxes_x2_noise_map
    ):

        point_dataset = al.PointDataset(
            name="point_0",
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
            fluxes=fluxes_x2,
            fluxes_noise_map=fluxes_x2_noise_map,
        )

        point_dict = al.PointDict(point_dataset_list=[point_dataset])

        model = af.Collection(
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5,
                    sis=al.mp.SphIsothermal(einstein_radius=1.0),
                    point_0=al.ps.PointFlux(flux=1.0),
                )
            )
        )

        solver = mock.MockPositionsSolver(model_positions=positions_x2)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        instance = model.instance_from_unit_vector([])

        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        fit_fluxes = al.FitFluxes(
            name="point_0",
            fluxes=fluxes_x2,
            noise_map=fluxes_x2_noise_map,
            positions=positions_x2,
            tracer=tracer,
        )

        assert (
            fit_positions.log_likelihood + fit_fluxes.log_likelihood
            == analysis_log_likelihood
        )

        model_positions = al.Grid2DIrregular([(0.0, 1.0), (1.0, 2.0)])
        solver = mock.MockPositionsSolver(model_positions=model_positions)

        analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

        instance = model.instance_from_unit_vector([])
        analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

        fit_positions = al.FitPositionsImage(
            name="point_0",
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=solver,
        )

        fit_fluxes = al.FitFluxes(
            name="point_0",
            fluxes=fluxes_x2,
            noise_map=fluxes_x2_noise_map,
            positions=positions_x2,
            tracer=tracer,
        )

        assert fit_positions.residual_map.in_list == [1.0, 1.0]
        assert fit_positions.chi_squared == 2.0
        assert (
            fit_positions.log_likelihood + fit_fluxes.log_likelihood
            == analysis_log_likelihood
        )
