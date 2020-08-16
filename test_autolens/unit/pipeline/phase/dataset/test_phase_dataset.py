from os import path

import autolens as al
import numpy as np
import pytest
from autolens import exc
from test_autolens import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__positions_are_input__are_used_in_analysis(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens dataset.

        imaging_7x7.positions = al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])
        phase_imaging_7x7.meta_dataset.settings.positions_threshold = 0.2

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (
            analysis.masked_dataset.positions.in_list[0][0] == np.array([1.0, 1.0])
        ).all()
        assert (
            analysis.masked_dataset.positions.in_list[0][1] == np.array([2.0, 2.0])
        ).all()
        assert analysis.masked_imaging.positions_threshold == 0.2

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):

            imaging_7x7.positions = None
            phase_imaging_7x7.meta_dataset.settings.positions_threshold = 0.2

            phase_imaging_7x7.make_analysis(
                dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
            )

            phase_imaging_7x7.meta_dataset.settings.positions_threshold = 0.2

            phase_imaging_7x7.make_analysis(
                dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
            )

    def test__positions_do_not_trace_within_threshold__raises_exception(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):

        imaging_7x7.positions = al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal),
                source=al.Galaxy(redshift=1.0),
            ),
            settings=al.PhaseSettingsImaging(positions_threshold=50.0),
            search=mock.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            ),
            settings=al.PhaseSettingsImaging(positions_threshold=0.0),
            search=mock.MockSearch(),
        )

        imaging_7x7.positions = al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer
            )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                source=al.Galaxy(redshift=1.0),
            ),
            settings=al.PhaseSettingsImaging(positions_threshold=0.5),
            search=mock.MockSearch(),
        )

        imaging_7x7.positions = al.GridCoordinates([[(1.0, 0.0), (-1.0, 0.0)]])

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.0)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=0.0)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer
            )

        imaging_7x7.positions = al.GridCoordinates(
            [[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        imaging_7x7.positions = al.GridCoordinates(
            [[(0.0, 0.0), (0.0, 0.0)], [(100.0, 0.0), (0.0, 0.0)]]
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.masked_dataset.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer
            )
