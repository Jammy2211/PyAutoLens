from os import path
import pytest

import autofit as af
import autolens as al

directory = path.dirname(path.realpath(__file__))


def test__tracer_for_instance(analysis_imaging_7x7):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                light=al.lp.SersicSph(intensity=2.0),
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            source=al.Galaxy(redshift=1.0),
        )
        + af.Collection(
            clump=al.Galaxy(
                redshift=0.5,
                light=al.lp.SersicSph(intensity=0.1),
                mass=al.mp.IsothermalSph(einstein_radius=0.2),
            )
        ),
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[0].redshift == 0.5
    assert tracer.galaxies[0].light.intensity == 2.0
    assert tracer.galaxies[0].mass.centre == pytest.approx((0.0, 0.0), 1.0e-4)
    assert tracer.galaxies[0].mass.einstein_radius == 1.0
    assert tracer.galaxies[2].redshift == 0.5
    assert tracer.galaxies[2].light.intensity == 0.1
    assert tracer.galaxies[2].mass.einstein_radius == 0.2


def test__tracer_for_instance__subhalo_redshift_rescale_used(analysis_imaging_7x7):
    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            subhalo=al.Galaxy(redshift=0.25, mass=al.mp.NFWSph(centre=(0.1, 0.2))),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[1].mass.centre == pytest.approx((0.1, 0.2), 1.0e-4)

    model = af.Collection(
        galaxies=af.Collection(
            lens=al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
            ),
            subhalo=al.Galaxy(redshift=0.75, mass=al.mp.NFWSph(centre=(0.1, 0.2))),
            source=al.Galaxy(redshift=1.0),
        )
    )

    instance = model.instance_from_unit_vector([])
    tracer = analysis_imaging_7x7.tracer_via_instance_from(instance=instance)

    assert tracer.galaxies[1].mass.centre == pytest.approx((-0.19959, -0.39919), 1.0e-4)