from os import path

import autolens as al
import pytest
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestFit:
    def test__fit_using_positions(
        self, positions_x2, positions_x2_noise_map, samples_with_result
    ):

        phase_positions_x2 = al.PhasePointSource(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            search=mock.MockSearch(samples=samples_with_result),
            positions_solver=mock.MockPositionsSolver(model_positions=positions_x2),
        )

        result = phase_positions_x2.run(
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
            results=mock.MockResults(),
        )
        assert isinstance(result.instance.galaxies[0], al.Galaxy)
        assert isinstance(result.instance.galaxies[0], al.Galaxy)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, positions_x2, positions_x2_noise_map
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.lp.EllipticalSersic(intensity=0.1)
        )

        phase_positions_x2 = al.PhasePointSource(
            galaxies=dict(lens=lens_galaxy),
            settings=al.SettingsPhasePositions(),
            search=mock.MockSearch(),
            positions_solver=mock.MockPositionsSolver(model_positions=positions_x2),
        )

        analysis = phase_positions_x2.make_analysis(
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
            results=mock.MockResults(),
        )
        instance = phase_positions_x2.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        positions_solver = mock.MockPositionsSolver(model_positions=positions_x2)

        fit_positions = al.FitPositionsImage(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit_positions.chi_squared == 0.0
        assert fit_positions.log_likelihood == fit_figure_of_merit

        model_positions = al.Grid2DIrregularGrouped([[(0.0, 1.0), (1.0, 2.0)]])
        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        phase_positions_x2 = al.PhasePointSource(
            galaxies=dict(lens=lens_galaxy),
            settings=al.SettingsPhasePositions(),
            search=mock.MockSearch(),
            positions_solver=positions_solver,
        )

        analysis = phase_positions_x2.make_analysis(
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
            results=mock.MockResults(),
        )
        instance = phase_positions_x2.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        fit_positions = al.FitPositionsImage(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit_positions.residual_map.in_grouped_list == [[1.0, 1.0]]
        assert fit_positions.chi_squared == 2.0
        assert fit_positions.log_likelihood == fit_figure_of_merit

    # def test__figure_of_merit__includes_fit_fluxes(
    #     self, positions_x2, positions_x2_noise_map, fluxes_x2, fluxes_x2_noise_map
    # ):
    #     lens_galaxy = al.Galaxy(
    #         redshift=0.5,
    #         sis=al.mp.SphericalIsothermal(einstein_radius=1.0),
    #         light=al.lp.PointSourceFlux(flux=1.0)
    #     )
    #
    #     phase_positions_x2 = al.PhasePointSource(
    #         galaxies=dict(lens=lens_galaxy),
    #         settings=al.SettingsPhasePositions(),
    #         search=mock.MockSearch(),
    #         positions_solver=mock.MockPositionsSolver(model_positions=positions_x2),
    #     )
    #
    #     print(positions_x2)
    #
    #     analysis = phase_positions_x2.make_analysis(
    #         positions=positions_x2,
    #         positions_noise_map=positions_x2_noise_map,
    #         fluxes=fluxes_x2,
    #         fluxes_noise_map=fluxes_x2_noise_map,
    #         results=mock.MockResults(),
    #     )
    #     instance = phase_positions_x2.model.instance_from_unit_vector([])
    #
    #     fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)
    #
    #     tracer = analysis.tracer_for_instance(instance=instance)
    #
    #     positions_solver = mock.MockPositionsSolver(model_positions=positions_x2)
    #
    #     fit_positions = al.FitPositionsImage(
    #         positions=positions_x2,
    #         noise_map=positions_x2_noise_map,
    #         tracer=tracer,
    #         positions_solver=positions_solver,
    #     )
    #
    #     fit_fluxes = al.FitFluxes(
    #         fluxes=fluxes_x2_noise_map,
    #         noise_map=fluxes_x2_noise_map,
    #         positions=positions_x2,
    #         tracer=tracer,
    #     )
    #
    #     assert fit_positions.log_likelihood + fit_fluxes.log_likelihood == fit_figure_of_merit

    # model_positions = al.Grid2DIrregularGrouped([[(0.0, 1.0), (1.0, 2.0)]])
    # positions_solver = mock.MockPositionsSolver(model_positions=model_positions)
    #
    # phase_positions_x2 = al.PhasePointSource(
    #     galaxies=dict(lens=lens_galaxy),
    #     settings=al.SettingsPhasePositions(),
    #     search=mock.MockSearch(),
    #     positions_solver=positions_solver,
    # )
    #
    # analysis = phase_positions_x2.make_analysis(
    #     positions=positions_x2,
    #     positions_noise_map=positions_x2_noise_map,
    #     results=mock.MockResults(),
    # )
    # instance = phase_positions_x2.model.instance_from_unit_vector([])
    # fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)
    #
    # fit_positions = al.FitPositionsImage(
    #     positions=positions_x2,
    #     noise_map=positions_x2_noise_map,
    #     tracer=tracer,
    #     positions_solver=positions_solver,
    # )
    #
    # assert fit_positions.residual_map.in_grouped_list == [[1.0, 1.0]]
    # assert fit_positions.chi_squared == 2.0
    # assert fit_positions.log_likelihood == fit_figure_of_merit
