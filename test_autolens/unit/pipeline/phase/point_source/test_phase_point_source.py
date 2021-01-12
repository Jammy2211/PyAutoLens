from os import path

import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__positions__settings_inputs_are_used_in_positions(
        self, positions_x2, positions_x2_noise_map
    ):
        phase_positions_x2 = al.PhasePointSource(
            settings=al.SettingsPhasePositions(),
            search=mock.MockSearch(),
            positions_solver=mock.MockPositionsSolver(model_positions=positions_x2),
        )

        assert isinstance(phase_positions_x2.settings, al.SettingsPhasePositions)

        analysis = phase_positions_x2.make_analysis(
            positions=positions_x2, positions_noise_map=positions_x2_noise_map
        )

        assert analysis.positions.in_grouped_list == positions_x2.in_grouped_list
        assert (
            analysis.noise_map.in_grouped_list == positions_x2_noise_map.in_grouped_list
        )

    def test___phase_info_is_made(
        self, phase_positions_x2, positions_x2, positions_x2_noise_map
    ):

        phase_positions_x2.make_analysis(
            positions=positions_x2,
            positions_noise_map=positions_x2_noise_map,
            results=mock.MockResults(),
        )

        file_phase_info = path.join(
            phase_positions_x2.search.paths.output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        search = phase_info.readline()
        cosmology = phase_info.readline()

        phase_info.close()

        assert search == "Optimizer = MockSearch \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )
