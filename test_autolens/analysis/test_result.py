import os
import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.analysis import result as res
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = os.path.dirname(os.path.realpath(__file__))


class TestResultAbstract:

    def test__max_log_likelihood_tracer_available_as_result(
        self, masked_imaging_7x7, samples_with_result
    ):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                lens=al.Galaxy(redshift=0.5,),
                source=al.Galaxy(redshift=1.0),
            )
        )

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
        )

        search = mock.MockSearch("test_phase_2", samples=samples_with_result)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result.max_log_likelihood_tracer, al.Tracer)
        assert result.max_log_likelihood_tracer.galaxies[0].light.intensity == 1.0
        assert result.max_log_likelihood_tracer.galaxies[1].light.intensity == 2.0

    def test__max_log_likelihood_tracer_source_light_profile_centres_correct(
        self, imaging_7x7, mask_7x7
    ):

        lens = al.Galaxy(redshift=0.5, light=al.lp.SphericalSersic(intensity=1.0))

        source = al.Galaxy(
            redshift=1.0, light=al.lp.SphericalSersic(centre=(1.0, 2.0), intensity=2.0)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.source_plane_light_profile_centre.in_list == [(1.0, 2.0)]

        source = al.Galaxy(
            redshift=1.0,
            light=al.lp.SphericalSersic(centre=(1.0, 2.0), intensity=2.0),
            light1=al.lp.SphericalSersic(centre=(3.0, 4.0), intensity=2.0),
        )

        source_1 = al.Galaxy(
            redshift=1.0, light=al.lp.SphericalSersic(centre=(5.0, 6.0), intensity=2.0)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source, source_1])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.source_plane_light_profile_centre.in_list == [(1.0, 2.0)]

        tracer = al.Tracer.from_galaxies(galaxies=[al.Galaxy(redshift=0.5)])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.source_plane_light_profile_centre == None

    def test__max_log_likelihood_tracer_source_inversion_centres_correct(
        self, imaging_7x7, mask_7x7
    ):
        lens = al.Galaxy(redshift=0.5, light=al.lp.SphericalSersic(intensity=1.0))

        source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.Rectangular((3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.max_log_likelihood_fit.inversion.reconstruction == pytest.approx(
            np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]), 1.0e-1
        )

        assert result.source_plane_inversion_centre.in_list == [(0.0, 0.0)]

        lens = al.Galaxy(redshift=0.5, light=al.lp.SphericalSersic(intensity=1.0))
        source = al.Galaxy(redshift=1.0)

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.source_plane_inversion_centre == None

    def test__max_log_likelihood_tracer_source_centres_correct(
        self, imaging_7x7, mask_7x7
    ):

        lens = al.Galaxy(redshift=0.5, light=al.lp.SphericalSersic(intensity=1.0))
        source = al.Galaxy(
            redshift=1.0,
            light=al.lp.SphericalSersic(centre=(9.0, 8.0), intensity=2.0),
            pixelization=al.pix.Rectangular((3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert result.source_plane_centre.in_list == [(0.0, 0.0)]

    def test__max_log_likelihood_tracer__multiple_image_positions_of_source_plane_centres_and_separations(
        self, imaging_7x7, mask_7x7
    ):

        lens = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.001, 0.001),
                einstein_radius=1.0,
                elliptical_comps=(0.0, 0.111111),
            ),
        )

        source = al.Galaxy(
            redshift=1.0,
            light=al.lp.SphericalSersic(centre=(0.0, 0.0), intensity=2.0),
            light1=al.lp.SphericalSersic(centre=(0.0, 0.1), intensity=2.0),
            pixelization=al.pix.Rectangular((3, 3)),
            regularization=al.reg.Constant(coefficient=1.0),
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens, source])

        samples = mock.MockSamples(max_log_likelihood_instance=tracer)

        phase_dataset_7x7 = al.PhaseImaging(
            search=mock.MockSearch("test_phase_2", samples=samples)
        )

        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        mask = al.Mask2D.unmasked(
            shape_native=(100, 100), pixel_scales=0.05, sub_size=1
        )

        result.analysis.masked_dataset.mask = mask

        multiple_images = (
            result.image_plane_multiple_image_positions_of_source_plane_centres
        )

        grid = al.Grid2D.from_mask(mask=mask)

        solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.001)

        multiple_images_manual = solver.solve(
            lensing_obj=tracer,
            source_plane_coordinate=result.source_plane_inversion_centre[0],
        )

        assert multiple_images.in_list[0] == multiple_images_manual.in_list[0]


class TestResultImaging:

    def test__result_imaging_is_returned(self, masked_imaging_7x7):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=al.Galaxy(redshift=0.5))
        )

        analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch(name="test_phase")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultImaging)

    def test___image_dict(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(redshift=0.5)
        galaxies.source = al.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
        )

        result = res.ResultImaging(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.lens = al.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "lens")].native == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)


class TestResultInterferometer:
    def test__result_interferometer_is_returned(self, masked_interferometer_7):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=al.Galaxy(redshift=0.5))
        )

        analysis = al.AnalysisInterferometer(dataset=masked_interferometer_7)

        search = mock.MockSearch(name="test_phase")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultInterferometer)