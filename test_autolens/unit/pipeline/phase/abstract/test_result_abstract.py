import os
from os import path
import numpy as np

import pytest

import autofit as af
import autolens as al
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/config/".format(directory))

    af.conf.instance = af.conf.Config("{}/config/".format(directory))


class TestGeneric:
    def test__results_of_phase_are_available_as_properties(self, imaging_7x7, mask_7x7):

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=[
                al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0))
            ],
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert isinstance(result, al.AbstractPhase.Result)


class TestTracer:
    def test__most_likely_tracer_available_as_result(self, imaging_7x7, mask_7x7):

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(
                    redshift=1.0, light=al.lp.EllipticalCoreSersic(intensity=2.0)
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert isinstance(result.most_likely_tracer, al.Tracer)
        assert result.most_likely_tracer.galaxies[0].light.intensity == 1.0
        assert result.most_likely_tracer.galaxies[1].light.intensity == 2.0

    def test__most_likely_tracer_source_light_profile_centres_correct(
        self, imaging_7x7, mask_7x7
    ):

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    light=al.lp.EllipticalCoreSersic(centre=(1.0, 2.0), intensity=2.0),
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert result.source_plane_light_profile_centres.in_list == [[(1.0, 2.0)]]

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    light=al.lp.EllipticalCoreSersic(centre=(1.0, 2.0), intensity=2.0),
                    light1=al.lp.EllipticalCoreSersic(centre=(3.0, 4.0), intensity=2.0),
                ),
                source1=al.Galaxy(
                    redshift=1.0,
                    light=al.lp.EllipticalCoreSersic(centre=(5.0, 6.0), intensity=2.0),
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert result.source_plane_light_profile_centres.in_list == [
            [(1.0, 2.0), (3.0, 4.0)],
            [(5.0, 6.0)],
        ]

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(lens=al.Galaxy(redshift=0.5)),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert result.source_plane_light_profile_centres == []

    def test__most_likely_tracer_source_inversion_centres_correct(
        self, imaging_7x7, mask_7x7
    ):

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    pixelization=al.pix.Rectangular((3, 3)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert result.most_likely_fit.inversion.reconstruction == pytest.approx(
            np.array(
                [
                    0.80707817,
                    0.80929118,
                    0.80707817,
                    0.80929118,
                    0.81251066,
                    0.80929118,
                    0.80707817,
                    0.80929118,
                    0.80707817,
                ]
            ),
            1.0e-4,
        )

        assert result.source_plane_inversion_centres.in_list == [[(0.0, 0.0)]]

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(redshift=1.0),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert result.source_plane_inversion_centres == []

    def test__most_likely_tracer_source_centres_correct(self, imaging_7x7, mask_7x7):

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5, light=al.lp.EllipticalSersic(intensity=1.0)
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    light=al.lp.EllipticalCoreSersic(centre=(9.0, 8.0), intensity=2.0),
                    pixelization=al.pix.Rectangular((3, 3)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert result.source_plane_centres.in_list == [[(9.0, 8.0), (0.0, 0.0)]]

    def test__most_likely_tracer__multiple_image_positions_of_source_plane_centres_and_separations(
        self, imaging_7x7, mask_7x7
    ):

        phase_dataset_7x7 = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.EllipticalIsothermal(
                        centre=(0.001, 0.001), einstein_radius=1.0, axis_ratio=0.8
                    ),
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    light=al.lp.EllipticalCoreSersic(centre=(0.0, 0.0), intensity=2.0),
                    light1=al.lp.EllipticalCoreSersic(centre=(0.0, 0.0), intensity=2.0),
                    pixelization=al.pix.Rectangular((3, 3)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            ),
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        # TODO : Again, we'll remove this need to pass a mask around when the Tracer uses an adaptive gird..

        result.analysis.masked_dataset.mask = al.Mask.unmasked(
            shape_2d=(100, 100), pixel_scales=0.05, sub_size=1
        )

        coordinates = (
            result.image_plane_multiple_image_positions_of_source_plane_centres
        )

        assert coordinates.in_list[0][0] == pytest.approx((1.025, -0.025), 1.0e-4)
        assert coordinates.in_list[0][1] == pytest.approx((0.025, -0.975), 1.0e-4)
        assert coordinates.in_list[0][2] == pytest.approx((0.025, 0.975), 1.0e-4)
        assert coordinates.in_list[0][3] == pytest.approx((-1.025, -0.025), 1.0e-4)
        assert coordinates.in_list[1][0] == pytest.approx((1.025, -0.025), 1.0e-4)
        assert coordinates.in_list[1][1] == pytest.approx((0.025, -0.975), 1.0e-4)
        assert coordinates.in_list[1][2] == pytest.approx((0.025, 0.975), 1.0e-4)
        assert coordinates.in_list[1][3] == pytest.approx((-1.025, -0.025), 1.0e-4)
        assert coordinates.in_list[2][0] == pytest.approx((0.225, -0.375), 1.0e-4)
        assert coordinates.in_list[2][1] == pytest.approx((-1.125, 1.025), 1.0e-4)
