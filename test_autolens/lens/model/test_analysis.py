import numpy as np
from os import path
import pytest

from autoconf import conf
import autofit as af
import autolens as al
from autolens import exc

directory = path.dirname(path.realpath(__file__))


def test__tracer_for_instance(analysis_imaging_7x7):

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                light=al.lp.SphSersic(intensity=2.0),
                mass=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source=al.Galaxy(redshift=1.0),
        ),
        clumps=af.Collection(
            clump=al.Galaxy(
                redshift=0.5,
                light=al.lp.SphSersic(intensity=0.1),
                mass=al.mp.SphIsothermal(einstein_radius=0.2),
            )
        ),
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[0].light.intensity == 2.0
    assert tracer.galaxies[0].mass.centre == pytest.approx((0.0, 0.0), 1.0e-4)
    assert tracer.galaxies[0].mass.einstein_radius == 1.0
    assert tracer.galaxies[1].redshift == 0.5
    assert tracer.galaxies[1].light.intensity == 0.1
    assert tracer.galaxies[1].mass.einstein_radius == 0.2


def test__tracer_for_instance__subhalo_redshift_rescale_used(analysis_imaging_7x7):

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            subhalo=al.Galaxy(redshift=0.25, mass=al.mp.SphNFW(centre=(0.1, 0.2))),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[0].mass.centre == pytest.approx((0.1, 0.2), 1.0e-4)

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            subhalo=al.Galaxy(redshift=0.75, mass=al.mp.SphNFW(centre=(0.1, 0.2))),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[1].mass.centre == pytest.approx((-0.19959, -0.39919), 1.0e-4)


def test__use_border__determines_if_border_pixel_relocation_is_used(
    self, masked_imaging_7x7
):

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5, mass=al.mp.SphIsothermal(einstein_radius=100.0)
            ),
            source=al.Galaxy(
                redshift=1.0,
                pixelization=al.pix.Rectangular(shape=(3, 3)),
                regularization=al.reg.Constant(coefficient=1.0),
            ),
        )
    )

    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=al.SettingsImaging(sub_size_inversion=2)
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        settings_pixelization=al.SettingsPixelization(use_border=True),
    )

    analysis.dataset.grid_inversion[4] = np.array([[500.0, 0.0]])

    instance = model.instance_from_unit_vector([])
    tracer = analysis.tracer_via_instance_from(instance=instance)
    fit = analysis.fit_imaging_via_tracer_from(
        tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
    )

    assert fit.inversion.linear_obj_list[0].source_grid_slim[4][0] == pytest.approx(
        97.19584, 1.0e-2
    )
    assert fit.inversion.linear_obj_list[0].source_grid_slim[4][1] == pytest.approx(
        -3.699999, 1.0e-2
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        settings_pixelization=al.SettingsPixelization(use_border=False),
    )

    analysis.dataset.grid_inversion[4] = np.array([300.0, 0.0])

    instance = model.instance_from_unit_vector([])
    tracer = analysis.tracer_via_instance_from(instance=instance)
    fit = analysis.fit_imaging_via_tracer_from(
        tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
    )

    assert fit.inversion.linear_obj_list[0].source_grid_slim[4][0] == pytest.approx(
        200.0, 1.0e-4
    )


def test__analysis_no_positions__removes_positions_and_threshold(
    self, masked_imaging_7x7
):

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]),
        settings_lens=al.SettingsLens(positions_threshold=0.01),
    )

    assert analysis.no_positions.positions == None
    assert analysis.no_positions.settings_lens.positions_threshold == None


def test__check_preloads(self, masked_imaging_7x7):

    conf.instance["general"]["test"]["check_preloads"] = True

    lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.EllSersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(lens=lens_galaxy))

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    instance = model.instance_from_unit_vector([])
    tracer = analysis.tracer_via_instance_from(instance=instance)
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    analysis.preloads.check_via_fit(fit=fit)

    analysis.preloads.blurred_image = fit.blurred_image

    analysis.preloads.check_via_fit(fit=fit)

    analysis.preloads.blurred_image = fit.blurred_image + 1.0

    with pytest.raises(exc.PreloadsException):
        analysis.preloads.check_via_fit(fit=fit)

    # conf.instance["general"]["test"]["check_preloads"] = False
    #
    # analysis.check_preloads(fit=fit)
