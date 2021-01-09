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

        phase_positions_x2 = al.PhasePositions(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic),
                source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
            ),
            search=mock.MockSearch(samples=samples_with_result),
            positions_solver=mock.MockPositionsSolver(model_positions=positions_x2),
        )

        result = phase_positions_x2.run(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
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

        phase_positions_x2 = al.PhasePositions(
            galaxies=dict(lens=lens_galaxy),
            settings=al.SettingsPhasePositions(),
            search=mock.MockSearch(),
            positions_solver=mock.MockPositionsSolver(model_positions=positions_x2),
        )

        analysis = phase_positions_x2.make_analysis(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            results=mock.MockResults(),
        )
        instance = phase_positions_x2.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        tracer = analysis.tracer_for_instance(instance=instance)

        positions_solver = mock.MockPositionsSolver(model_positions=positions_x2)

        fit = al.FitPositionsImagePlane(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit.chi_squared == 0.0
        assert fit.log_likelihood == fit_figure_of_merit

        model_positions = al.GridIrregularGrouped([[(0.0, 1.0), (1.0, 2.0)]])
        positions_solver = mock.MockPositionsSolver(model_positions=model_positions)

        phase_positions_x2 = al.PhasePositions(
            galaxies=dict(lens=lens_galaxy),
            settings=al.SettingsPhasePositions(),
            search=mock.MockSearch(),
            positions_solver=positions_solver,
        )

        analysis = phase_positions_x2.make_analysis(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            results=mock.MockResults(),
        )
        instance = phase_positions_x2.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        fit = al.FitPositionsImagePlane(
            positions=positions_x2,
            noise_map=positions_x2_noise_map,
            tracer=tracer,
            positions_solver=positions_solver,
        )

        assert fit.residual_map.in_grouped_list == [[1.0, 1.0]]
        assert fit.chi_squared == 2.0
        assert fit.log_likelihood == fit_figure_of_merit
