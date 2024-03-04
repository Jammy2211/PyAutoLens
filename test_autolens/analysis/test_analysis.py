import numpy as np
from os import path
import os
import pytest

from autoconf import conf
from autoconf.dictable import from_json

import autofit as af
import autolens as al
from autolens import exc

directory = path.dirname(path.realpath(__file__))


def test__tracer_instance__subhalo_redshift_rescale_used(analysis_imaging_7x7):
    model = af.Collection(
        tracer=af.Model(
            al.Tracer,
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
                ),
                subhalo=al.Galaxy(redshift=0.25, mass=al.mp.NFWSph(centre=(0.1, 0.2))),
                source=al.Galaxy(redshift=1.0),
            ),
        )
    )

    instance = model.instance_from_unit_vector([])

    assert instance.galaxies[0].mass.centre == pytest.approx((0.1, 0.2), 1.0e-4)

    model = af.Collection(
        tracer=af.Model(
            al.Tracer,
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
                ),
                subhalo=al.Galaxy(redshift=0.75, mass=al.mp.NFWSph(centre=(0.1, 0.2))),
                source=al.Galaxy(redshift=1.0),
            ),
        )
    )

    instance = model.instance_from_unit_vector([])

    assert instance.galaxies[1].mass.centre == pytest.approx(
        (-0.19959, -0.39919), 1.0e-4
    )


def test__relocate_pix_border__determines_if_border_pixel_relocation_is_used(
    masked_imaging_7x7,
):
    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    model = af.Collection(
        tracer=af.Model(
            al.Tracer,
            galaxies=af.Collection(
                lens=al.Galaxy(
                    redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=100.0)
                ),
                source=al.Galaxy(redshift=1.0, pixelization=pixelization),
            ),
        )
    )

    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=al.SettingsImaging(sub_size_pixelization=2)
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        settings_inversion=al.SettingsInversion(relocate_pix_border=True),
    )

    analysis.dataset.grid_pixelization[4] = np.array([[500.0, 0.0]])

    instance = model.instance_from_unit_vector([])
    fit = analysis.fit_from(instance=instance)

    assert fit.inversion.linear_obj_list[0].source_plane_data_grid[4][
        0
    ] == pytest.approx(97.19584, 1.0e-2)
    assert fit.inversion.linear_obj_list[0].source_plane_data_grid[4][
        1
    ] == pytest.approx(-3.699999, 1.0e-2)

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7,
        settings_inversion=al.SettingsInversion(relocate_pix_border=False),
    )

    analysis.dataset.grid_pixelization[4] = np.array([300.0, 0.0])

    instance = model.instance_from_unit_vector([])
    fit = analysis.fit_from(
        instance=instance,
    )

    assert fit.inversion.linear_obj_list[0].source_plane_data_grid[4][
        0
    ] == pytest.approx(200.0, 1.0e-4)


def test__modify_before_fit__inversion_no_positions_likelihood__raises_exception(
    masked_imaging_7x7,
):
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph())

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(), regularization=al.reg.Constant()
    )

    source = al.Galaxy(redshift=1.0, pixelization=pixelization)

    model = af.Collection(
        tracer=af.Model(al.Tracer, galaxies=af.Collection(lens=lens, source=source))
    )

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    with pytest.raises(exc.AnalysisException):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    positions_likelihood = al.PositionsLHPenalty(
        positions=al.Grid2DIrregular([(1.0, 100.0), (200.0, 2.0)]), threshold=0.01
    )

    analysis = al.AnalysisImaging(
        dataset=masked_imaging_7x7, positions_likelihood=positions_likelihood
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)


def test__check_preloads(masked_imaging_7x7):
    conf.instance["general"]["test"]["check_preloads"] = True

    lens_galaxy = al.Galaxy(redshift=0.5, light=al.lp.Sersic(intensity=0.1))

    model = af.Collection(
        tracer=af.Model(al.Tracer, galaxies=af.Collection(lens=lens_galaxy))
    )

    analysis = al.AnalysisImaging(dataset=masked_imaging_7x7)

    instance = model.instance_from_unit_vector([])
    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=instance)

    analysis.preloads.check_via_fit(fit=fit)

    analysis.preloads.blurred_image = fit.blurred_image

    analysis.preloads.check_via_fit(fit=fit)

    analysis.preloads.blurred_image = fit.blurred_image + 1.0

    with pytest.raises(exc.PreloadsException):
        analysis.preloads.check_via_fit(fit=fit)


def test__save_results__tracer_output_to_json(analysis_imaging_7x7):
    lens = al.Galaxy(redshift=0.5)
    source = al.Galaxy(redshift=1.0)

    model = af.Collection(
        tracer=af.Model(al.Tracer, galaxies=af.Collection(lens=lens, source=source))
    )

    tracer = al.Tracer(galaxies=[lens, source])

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths,
        result=al.m.MockResult(max_log_likelihood_tracer=tracer, model=model),
    )

    tracer = from_json(file_path=paths._files_path / "tracer.json")

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[1].redshift == 1.0

    os.remove(paths._files_path / "tracer.json")
