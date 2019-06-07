import numpy as np
import pytest
from astropy import cosmology as cosmo

import autolens.pipeline.phase.phase_imaging
from autofit.mapper import model
from autofit.mapper import model_mapper as mm
from autofit.tools import pipeline as pl
from autolens.lens import ray_tracing as rt
from autolens.model import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline.phase import phase as ph


@pytest.fixture(name="lens_galaxy")
def make_lens_galaxy():
    return g.Galaxy(redshift=1.0, light=lp.SphericalSersic(), mass=mp.SphericalIsothermal())


@pytest.fixture(name="source_galaxy")
def make_source_galaxy():
    return g.Galaxy(redshift=2.0, light=lp.SphericalSersic())


@pytest.fixture(name="lens_galaxies")
def make_lens_galaxies(lens_galaxy):
    lens_galaxies = model.ModelInstance()
    lens_galaxies.lens = lens_galaxy
    return lens_galaxies


@pytest.fixture(name="all_galaxies")
def make_all_galaxies(lens_galaxy, source_galaxy):
    galaxies = model.ModelInstance()
    galaxies.lens = lens_galaxy
    galaxies.source = source_galaxy
    return galaxies


@pytest.fixture(name="lens_instance")
def make_lens_instance(lens_galaxies):
    instance = model.ModelInstance()
    instance.lens_galaxies = lens_galaxies
    return instance


@pytest.fixture(name="lens_result")
def make_lens_result(lens_data_5x5, lens_instance):
    return autolens.pipeline.phase.phase_imaging.LensPlanePhase.Result(
        constant=lens_instance, figure_of_merit=1.0, previous_variable=mm.ModelMapper(), gaussian_tuples=None,
        analysis=autolens.pipeline.phase.phase_imaging.LensPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=1.0), optimizer=None)


@pytest.fixture(name="lens_source_instance")
def make_lens_source_instance(lens_galaxy, source_galaxy):

    source_galaxies = model.ModelInstance()
    lens_galaxies = model.ModelInstance()
    source_galaxies.source = source_galaxy
    lens_galaxies.lens = lens_galaxy

    instance = model.ModelInstance()
    instance.source_galaxies = source_galaxies
    instance.lens_galaxies = lens_galaxies
    return instance


@pytest.fixture(name="lens_source_result")
def make_lens_source_result(lens_data_5x5, lens_source_instance):
    return autolens.pipeline.phase.phase_imaging.LensSourcePlanePhase.Result(
        constant=lens_source_instance, figure_of_merit=1.0, previous_variable=mm.ModelMapper(), gaussian_tuples=None,
        analysis=autolens.pipeline.phase.phase_imaging.LensSourcePlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=1.0), optimizer=None)


@pytest.fixture(name="multi_plane_instance")
def make_multi_plane_instance(all_galaxies):
    instance = model.ModelInstance()
    instance.galaxies = all_galaxies
    return instance


@pytest.fixture(name="multi_plane_result")
def make_multi_plane_result(lens_data_5x5, multi_plane_instance):
    return autolens.pipeline.phase.phase_imaging.MultiPlanePhase.Result(
        constant=multi_plane_instance, figure_of_merit=1.0, previous_variable=mm.ModelMapper(), gaussian_tuples=None,
        analysis=autolens.pipeline.phase.phase_imaging.MultiPlanePhase.Analysis(
            lens_data=lens_data_5x5, cosmology=cosmo.Planck15, positions_threshold=1.0), optimizer=None)


class TestImagePassing(object):

    def test_lens_galaxy_dict(self, lens_result, lens_galaxy):

        assert lens_result.name_galaxy_tuples == [("lens_galaxies_lens", lens_galaxy)]

    def test_lens_source_galaxy_dict(self, lens_source_result, lens_galaxy, source_galaxy):
        assert lens_source_result.name_galaxy_tuples == [
            ("source_galaxies_source", source_galaxy),
            ("lens_galaxies_lens", lens_galaxy)
        ]

    def test_multi_plane_galaxy_dict(self, multi_plane_result, lens_galaxy, source_galaxy):
        assert multi_plane_result.name_galaxy_tuples == [
            ("galaxies_lens", lens_galaxy),
            ("galaxies_source", source_galaxy)
        ]

    def test_lens_image_dict(self, lens_result):
        image_dict = lens_result.image_dict
        assert isinstance(image_dict["lens_galaxies_lens"], np.ndarray)

    def test_lens_source_image_dict(self, lens_source_result):
        image_dict = lens_source_result.image_dict
        assert isinstance(image_dict["lens_galaxies_lens"], np.ndarray)
        assert isinstance(image_dict["source_galaxies_source"], np.ndarray)

    def test_multi_plane_image_dict(self, multi_plane_result):
        image_dict = multi_plane_result.image_dict
        assert isinstance(image_dict["galaxies_lens"], np.ndarray)
        assert isinstance(image_dict["galaxies_source"], np.ndarray)

    def test_galaxy_image_dict(self, lens_galaxy, source_galaxy, grid_stack_5x5):
        tracer = rt.TracerImageSourcePlanes([lens_galaxy], [source_galaxy], grid_stack_5x5)

        assert len(tracer.galaxy_image_dict) == 2
        assert lens_galaxy in tracer.galaxy_image_dict
        assert source_galaxy in tracer.galaxy_image_dict

    def test_associate_images_lens(self, lens_instance, lens_result, lens_data_5x5):
        results_collection = pl.ResultsCollection()
        results_collection.add("phase", lens_result)
        phase = autolens.pipeline.phase.phase_imaging.LensPlanePhase.Analysis(lens_data_5x5, None, None, results_collection)

        instance = phase.associate_images(lens_instance)

        assert (instance.lens_galaxies.lens.image == lens_result.image_dict["lens_galaxies_lens"]).all()

    def test_associate_images_lens_source(self, lens_source_instance, lens_source_result, lens_data_5x5):
        results_collection = pl.ResultsCollection()
        results_collection.add("phase", lens_source_result)
        phase = autolens.pipeline.phase.phase_imaging.LensSourcePlanePhase.Analysis(lens_data_5x5, None, None, results_collection)

        instance = phase.associate_images(lens_source_instance)

        assert (instance.lens_galaxies.lens.image == lens_source_result.image_dict["lens_galaxies_lens"]).all()
        assert (instance.source_galaxies.source.image == lens_source_result.image_dict["source_galaxies_source"]).all()

    def test_associate_images_multi_plane(self, multi_plane_instance, multi_plane_result, lens_data_5x5):
        results_collection = pl.ResultsCollection()
        results_collection.add("phase", multi_plane_result)
        phase = autolens.pipeline.phase.phase_imaging.MultiPlanePhase.Analysis(lens_data_5x5, None, None, results_collection)

        instance = phase.associate_images(multi_plane_instance)

        assert (instance.galaxies.lens.image == multi_plane_result.image_dict["galaxies_lens"]).all()
        assert (instance.galaxies.source.image == multi_plane_result.image_dict["galaxies_source"]).all()
