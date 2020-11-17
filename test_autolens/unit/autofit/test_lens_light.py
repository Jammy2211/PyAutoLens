# import pytest
#
# import autofit as af
# import autolens as al
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
# @pytest.fixture(name="collection_7")
# def make_collection_7(collection_6, phase6):
#     collection_6.add_phase(phase6)
#     return collection_6
#
# @pytest.fixture(name="collection_8")
# def make_collection_8(collection_8, phase7):
#     collection_8.add_phase(phase7)
#     return collection_8
#
# @pytest.fixture(name="phase1")
# def make_phase_1():
#
#     lens = al.GalaxyModel(
#         redshift=redshift_lens,
#         bulge=al.lp.EllipticalSersic,
#         disk=al.lp.EllipticalExponential,
#     )
#
#     lens.bulge.centre = lens.disk.centre
#
#     return al.PhaseImaging(
#         name="phase[1]", galaxies=dict(lens=lens), search=af.DynestyStatic()
#     )
#
#
# def test__phase_1(phase1):
#
#     # 7 Bulge + 4 Disk (aligned centres)
#     assert phase1.model.prior_count == 11
#
#
# @pytest.fixture(name="phase2")
# def make_phase_2(phase1):
#
#     return al.PhaseImaging(
#         name="phase[2]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens,
#                 bulge=phase1.result.instance.galaxies.lens.bulge,
#                 disk=phase1.result.instance.galaxies.lens.disk,
#                 mass=al.mp.EllipticalIsothermal,
#                 hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 sersic=al.lp.EllipticalSersic,
#                 hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=af.DynestyStatic(),
#     )
#
#
# def test_phase_2(phase2):
#
#     # 5 Lens SIE + 12 Source Sersic
#     assert phase2.model.prior_count == 12
#
#
# @pytest.fixture(name="phase3")
# def make_phase_3(phase2):
#
#     lens = al.GalaxyModel(
#         redshift=redshift_lens,
#         bulge=al.lp.EllipticalSersic,
#         disk=al.lp.EllipticalExponential,
#         mass=phase2.result.instance.galaxies.lens.mass,
#         hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#     )
#
#     lens.bulge.centre = lens.disk.centre
#
#     return al.PhaseImaging(
#         name="phase[3]",
#         galaxies=dict(
#             lens=lens,
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 sersic=phase2.result.instance.galaxies.source.sersic,
#                 hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=af.DynestyStatic(),
#     )
#
#
# def test_phase_3(phase3):
#
#     # 7 Bulge + 4 Disk (aligned centres)
#     assert phase3.model.prior_count == 11
#
# @pytest.fixture(name="phase4")
# def make_phase_4(phase2, phase3):
#
#     return al.PhaseImaging(
#         name="phase[4]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens,
#                 bulge=phase3.result.model.galaxies.lens.bulge,
#                 disk=phase3.result.model.galaxies.lens.disk,
#                 mass=phase2.result.model.galaxies.lens.mass,
#                 hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 sersic=phase2.result.model.galaxies.source.sersic,
#                 hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=af.DynestyStatic(),
#     )
#
#
# def test_phase_4(phase4):
#
#     # 7 Bulge + 4 Disk (aligned centres) + 5 Lens SIE + 7 Source Sersic
#     assert phase4.model.prior_count == 23
#
# @pytest.fixture(name="phase5")
# def make_phase_5():
#
#     return al.PhaseImaging(
#         name="phase[5]",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens,
#                 bulge=af.last.instance.galaxies.lens.bulge,
#                 disk=af.last.instance.galaxies.lens.disk,
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
#         search=af.DynestyStatic(),
#     )
#
# def test_phase_5(phase5):
#
#     # 3 Source Inversion
#     assert phase5.model.prior_count == 3
#
# @pytest.fixture(name="phase6")
# def make_phase_6(phase5):
#
#     return al.PhaseImaging(
#         name="phase_6",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens,
#                 bulge=af.last[-1].instance.galaxies.lens.bulge,
#                 disk=af.last[-1].instance.galaxies.lens.disk,
#                 mass=af.last[-1].model.galaxies.lens.mass,
#                 shear=af.last[-1].model.galaxies.lens.shear,
#                 hyper_galaxy=phase5.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=phase5.result.instance.galaxies.source.pixelization,
#                 regularization=phase5.result.instance.galaxies.source.regularization,
#                 hyper_galaxy=phase5.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase5.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase5.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=af.DynestyStatic(),
#     )
#
# def test_phase_6(phase6):
#
#     # 5 Lens SIE
#     assert phase6.model.prior_count == 5
#
# @pytest.fixture(name="phase7")
# def make_phase_7(phase6):
#
#     return al.PhaseImaging(
#         name="phase_7",
#         galaxies=dict(
#             lens=al.GalaxyModel(
#                 redshift=redshift_lens,
#                 bulge=phase6.result.instance.galaxies.lens.bulge,
#                 disk=phase6.result.instance.galaxies.lens.disk,
#                 mass=phase6.result.instance.galaxies.lens.mass,
#                 shear=phase6.result.instance.galaxies.lens.shear,
#                 hyper_galaxy=phase6.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#             ),
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=al.pix.VoronoiBrightnessImage,
#                 regularization=al.reg.AdaptiveBrightness,
#                 hyper_galaxy=phase6.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase6.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase6.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=af.DynestyStatic(),
#     )
#
# def test_phase_7(phase7):
#
#     # 6 Source Inversion
#     assert phase7.model.prior_count == 6
#
# @pytest.fixture(name="phase8")
# def make_phase_8(phase6, phase7):
#
#     mass = af.PriorModel(al.mp.EllipticalIsothermal)
#     mass.axis_ratio = phase6.result.model.galaxies.lens.mass.axis_ratio
#     mass.phi = phase6.result.model.galaxies.lens.mass.phi
#     mass.einstein_radius = phase6.result.model.galaxies.lens.mass.einstein_radius
#
#     lens = al.GalaxyModel(
#         redshift=redshift_lens,
#         bulge=phase6.result.instance.galaxies.lens.bulge,
#         disk=phase6.result.instance.galaxies.lens.disk,
#         mass=mass,
#         shear=phase6.result.model.galaxies.lens.shear,
#         hyper_galaxy=phase7.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#     )
#
#     return al.PhaseImaging(
#         name="phase_8",
#         galaxies=dict(
#             lens=lens,
#             source=al.GalaxyModel(
#                 redshift=redshift_source,
#                 pixelization=phase7.result.instance.galaxies.source.pixelization,
#                 regularization=phase7.result.instance.galaxies.source.regularization,
#                 hyper_galaxy=phase7.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
#             ),
#         ),
#         hyper_image_sky=phase7.result.hyper_combined.instance.optional.hyper_image_sky,
#         hyper_background_noise=phase7.result.hyper_combined.instance.optional.hyper_background_noise,
#         search=af.DynestyStatic(),
#     )
#
# def test_phase_8(phase8):
#
#     # 5 SIE
#     assert phase8.model.prior_count == 5
#
#
#
#
#
#
#
#
#
#
