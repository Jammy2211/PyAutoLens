# import pytest
#
# import autofit as af
# import autolens as al
# from autofit.non_linear.mock_search import MockSearch
#
# redshift_lens = (0.5,)
# redshift_source = (1.0,)
#
#
# class MockResult:
#     def __init__(self, model, instance):
#         self.model = model
#         self.instance = instance
#
#
# class MagicResultsCollection(af.ResultsCollection):
#     def add_phase(self, phase):
#         model = phase.model.populate(self)
#         self.add(
#             phase.name, MockResult(model, model.instance_from_prior_medians())
#         )
#
#
# @pytest.fixture(name="collection_1")
# def make_collection_1():
#     return MagicResultsCollection()
#
#
# @pytest.fixture(name="collection_2")
# def make_collection_2(collection_1, phase1):
#     collection_1.add_phase(phase1)
#     return collection_1
#
#
# @pytest.fixture(name="collection_3")
# def make_collection_3(collection_2, phase2):
#     collection_2.add_phase(phase2)
#     return collection_2
#
#
# @pytest.fixture(name="collection_4")
# def make_collection_4(collection_3, phase3):
#     collection_3.add_phase(phase3)
#     return collection_3
#
#
# @pytest.fixture(name="collection_5")
# def make_collection_5(collection_4, phase4):
#     collection_4.add_phase(phase4)
#     return collection_4
#
#
# @pytest.fixture(name="collection_6")
# def make_collection_6(collection_5, phase5):
#     collection_5.add_phase(phase5)
#     return collection_5
#
#
# @pytest.fixture(name="phase1")
# def make_phase_1():
#     return al.PhaseImaging(
#         name="phase[1]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens, mass=al.mp.EllipticalIsothermal
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source, sersic=al.lp.EllipticalSersic
#             ),
#         ),
#         search=MockSearch,
#     )
#
#
# def test_phase_1(phase1):
#     # 5 Lens SIE + 12 Source Sersic
#     assert phase1.model.prior_count == 12
#
#
# @pytest.fixture(name="phase2")
# def make_phase_2():
#     return al.PhaseImaging(
#         name="phase[2]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens,
#                 mass=af.last.instance.galaxies.lens.mass,
#                 hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=al.pix.VoronoiMagnification,
#                 regularization=al.reg.Constant,
#                 hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
#         search=MockSearch,
#     )
#
#
# def test_phase_2(collection_2, phase2):
#     # 3 Source Inversion
#     assert phase2.model.populate(collection_2).prior_count == 3
#
#
# @pytest.fixture(name="phase3")
# def make_phase_3(phase2):
#     return al.PhaseImaging(
#         name="phase[3]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens, mass=af.last[-1].model.galaxies.lens.mass
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=phase2.result.instance.galaxies.source.pixelization,
#                 regularization=phase2.result.instance.galaxies.source.regularization,
#                 hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         search=MockSearch,
#     )
#
#
# def test_phase_3(collection_3, phase3):
#     # 5 Lens SIE
#     assert phase3.model.populate(collection_3).prior_count == 5
#
#
# @pytest.fixture(name="phase4")
# def make_phase_phase_4(phase2, phase3):
#     return al.PhaseImaging(
#         name="phase[4]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens, mass=phase2.result.instance.galaxies.lens.mass
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=al.pix.VoronoiBrightnessImage,
#                 regularization=al.reg.AdaptiveBrightness,
#                 hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=MockSearch,
#     )
#
#
# def test_phase_4(collection_4, phase4):
#     # 6 Source Inversion
#     assert phase4.model.populate(collection_4).prior_count == 6
#
#
# @pytest.fixture(name="phase5")
# def make_phase_5(phase3, phase4):
#     return al.PhaseImaging(
#         name="phase[5]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens, mass=phase3.result.model.galaxies.lens.mass
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=phase4.result.instance.galaxies.source.pixelization,
#                 regularization=phase4.result.instance.galaxies.source.regularization,
#                 hyper_galaxy=phase4.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase4.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase4.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=MockSearch,
#     )
#
#
# def test_phase_5(collection_5, phase5):
#     # 5 Lens SIE
#     assert phase5.model.populate(collection_5).prior_count == 5
#
#
# @pytest.fixture(name="phase6")
# def make_phase_6():
#     mass = af.PriorModel(al.mp.EllipticalPowerLaw)
#
#     mass.centre = af.last.model.galaxies.lens.mass.centre
#     mass.axis_ratio = af.last.model.galaxies.lens.mass.axis_ratio
#     mass.phi = af.last.model.galaxies.lens.mass.phi
#     mass.einstein_radius = af.last.model_absolute(
#         a=0.3
#     ).galaxies.lens.mass.einstein_radius
#
#     source = al.GalaxyModel(
#         redshift=af.last.instance.galaxies.source.redshift,
#         pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
#         regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
#     )
#
#     return al.PhaseImaging(
#         name="phase_6",
#         galaxies=dict(
#             lens=al.GalaxyModel(redshift=redshift_lens, mass=mass), source=source
#         ),
#         hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
#         search=MockSearch,
#     )
#
#
# def test_phase_6(collection_6, phase6):
#     # 6 Lens SPLE
#     assert phase6.model.populate(collection_6).prior_count == 6
