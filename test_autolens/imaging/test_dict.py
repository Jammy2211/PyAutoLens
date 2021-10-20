import pytest
from astropy import cosmology as cosmo

import autolens as al


@pytest.fixture(
    name="tracer"
)
def make_tracer():
    mass = al.mp.EllIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

    disk = al.lp.EllExponential(
        centre=(0.3, 0.2),
        elliptical_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, disk=disk)

    return al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
    )


@pytest.fixture(
    name="tracer_dict"
)
def make_tracer_dict():
    return {
        'cosmology': {
            'H0': {'type': 'astropy.units.quantity.Quantity'},
            'Neff': 3.046,
            'Ob0': 0.0486,
            'Om0': 0.3075,
            'Tcmb0': {'type': 'astropy.units.quantity.Quantity'},
            'm_nu': {'type': 'astropy.units.quantity.Quantity'},
            'name': 'Planck15',
            'type': 'astropy.cosmology.core.FlatLambdaCDM'
        },
        'planes': [
            {
                'galaxies': [{
                    'hyper_galaxy': None,
                    'pixelization': None,
                    'redshift': 0.5,
                    'regularization': None,
                    'type': 'autogalaxy.galaxy.galaxy.Galaxy'
                }],
                'profiling_dict': None,
                'redshift': 0.5,
                'type': 'autogalaxy.plane.plane.Plane'
            },
            {
                'galaxies': [{
                    'hyper_galaxy': None,
                    'pixelization': None,
                    'redshift': 1.0,
                    'regularization': None,
                    'type': 'autogalaxy.galaxy.galaxy.Galaxy'
                }],
                'profiling_dict': None,
                'redshift': 1.0,
                'type': 'autogalaxy.plane.plane.Plane'
            }],
        'profiling_dict': None,
        'type': 'autolens.lens.ray_tracing.Tracer'
    }


def test_to_dict(
        tracer,
        tracer_dict
):
    assert tracer.dict() == tracer_dict


def test_from_dict(
        tracer,
        tracer_dict
):
    assert tracer.from_dict(
        tracer_dict
    ) == tracer
