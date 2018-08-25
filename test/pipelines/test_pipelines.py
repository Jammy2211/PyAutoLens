# from pipelines.lens_and_source import initializer
# from autolens.pipeline import phase as ph
# from autolens.autopipe import model_mapper
# from autolens.imaging import image as im
# from autolens.profiles import light_profiles
# from autolens.profiles import mass_profiles
# from autolens.analysis import galaxy_prior as gp
# from autolens.analysis import galaxy as g
# import numpy as np
# import pytest
#
# shape = (3, 3)
# shape1d = (9)
# pixel_scale = 0.2
#
# class MockConvolver(object):
#
#     def __init__(self):
#
#         pass
#
#     def convolve_image(self, image_plane_image, image_plane_blurring_image):
#         return np.ones(shape1d)
#
#
# class MockMaskedImage(np.ndarray):
#
#     def __new__(cls, array, *args, **kwargs):
#         mi = np.array(array, dtype='float64').view(cls)
#         mi.convolver_image = MockConvolver()
#         mi.pixel_scale = pixel_scale
#         return mi
#
#     def map_to_2d(self, image):
#         return np.ones(shape1d)
#
# class MockTracer(object):
#
#     def __init__(self):
#         pass
#
#     @property
#     def image_plane_image(self):
#         return np.ones(shape1d)
#
#     @property
#     def image_plane_images_of_galaxies(self):
#         return [np.ones(shape1d)]
#
#     @property
#     def image_plane_blurring_image(self):
#         return np.ones(shape1d)
#
#     @property
#     def image_plane_blurring_images_of_galaxies(self):
#         return [np.ones(shape1d)]
#
#
# class MockFitter(object):
#
#     def __init__(self):
#         pass
#
#
# class MockAnalysis(object):
#
#     def __init__(self):
#         self.masked_image = MockMaskedImage(array=np.ones((1,1)))
#
#     def tracer_for_instance(self, instance):
#         return MockTracer()
#
#
# @pytest.fixture(name="initializer_pipeline")
# def make_initializer_pipeline():
#     return initializer.make()
#
#
# @pytest.fixture(name="image")
# def make_image():
#     return im.Image(np.ones(shape), pixel_scale=pixel_scale, noise=np.ones(shape), psf=im.PSF(np.ones((3, 3))))
#
#
# @pytest.fixture(name="results_1")
# def make_results_1():
#     const = model_mapper.ModelInstance()
#     var = model_mapper.ModelMapper()
#     const.lens_galaxy = g.Galaxy(elliptical_sersic=light_profiles.EllipticalSersicLP())
#     var.lens_galaxy = gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersicLP)
#     return ph.LensProfilePhase.Result(constant=const, likelihood=1, variable=var, analysis=MockAnalysis())
#
#
# @pytest.fixture(name="results_2")
# def make_results_2():
#     const = model_mapper.ModelInstance()
#     var = model_mapper.ModelMapper()
#     var.lens_galaxy = gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermalMP)
#     var.source_galaxy = gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersicLP)
#     const.lens_galaxy = g.Galaxy(sie=mass_profiles.SphericalIsothermalMP())
#     const.source_galaxy = g.Galaxy(elliptical_sersic=light_profiles.EllipticalSersicLP())
#     return ph.LensMassAndSourceProfilePhase.Result(constant=const, likelihood=1, variable=var,
#                                                    analysis=MockAnalysis())
#
#
# @pytest.fixture(name="results_3")
# def make_results_3():
#     const = model_mapper.ModelInstance()
#     var = model_mapper.ModelMapper()
#     var.lens_galaxy = gp.GalaxyPrior(sie=mass_profiles.SphericalIsothermalMP,
#                                      elliptical_sersic=light_profiles.EllipticalSersicLP)
#     var.source_galaxy = gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersicLP)
#     const.lens_galaxy = g.Galaxy(sie=mass_profiles.SphericalIsothermalMP(),
#                                  elliptical_sersic=light_profiles.EllipticalSersicLP())
#     const.source_galaxy = g.Galaxy(elliptical_sersic=light_profiles.EllipticalSersicLP())
#     return ph.LensMassAndSourceProfilePhase.Result(constant=const, likelihood=1, variable=var,
#                                                    analysis=MockAnalysis(number_galaxies=2, shape=shape, value=0.5))
#
#
# class TestInitializationPipeline(object):
#
#     def test_phase1(self, initializer_pipeline, image):
#
#         phase1 = initializer_pipeline.phases[0]
#         analysis = phase1.make_analysis(image)
#
#         assert isinstance(phase1.lens_galaxies[0], gp.GalaxyPrior)
#
#         assert analysis.masked_image == np.ones((716,))
#         assert analysis.masked_image.sub_grid_size == 1
#         assert analysis.previous_results is None
#
#     def test_phase2(self, initializer_pipeline, image, results_1):
#
#         phase2 = initializer_pipeline.phases[1]
#         previous_results = ph.ResultsCollection([results_1])
#         analysis = phase2.make_analysis(image, previous_results)
#
#         assert analysis.masked_image == np.full((704,), 0.5)
#
#         assert isinstance(phase2.lens_galaxies[0], gp.GalaxyPrior)
#         assert isinstance(phase2.source_galaxies[0], gp.GalaxyPrior)
#         assert phase2.lens_galaxy.sie.centre == previous_results.first.variable.lens_galaxy.elliptical_sersic.centre
#
#     def test_phase3(self, profile_only_pipeline, image, results_1, results_2):
#         phase3 = profile_only_pipeline.phases[2]
#         previous_results = ph.ResultsCollection([results_1, results_2])
#
#         analysis = phase3.make_analysis(image, previous_results)
#
#         assert isinstance(phase3.lens_galaxy, gp.GalaxyPrior)
#         assert isinstance(phase3.source_galaxy, gp.GalaxyPrior)
#
#         assert analysis.masked_image == np.ones((716,))
#
#         assert phase3.lens_galaxy.elliptical_sersic == results_1.variable.lens_galaxy.elliptical_sersic
#         assert phase3.lens_galaxy.sie == results_2.variable.lens_galaxy.sie
#         assert phase3.source_galaxy == results_2.variable.source_galaxy