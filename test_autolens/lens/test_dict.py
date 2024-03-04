import pytest

import autolens as al
from autoconf.dictable import to_dict, from_dict


@pytest.fixture(name="tracer")
def make_tracer():
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

    disk = al.lp.Exponential(
        centre=(0.3, 0.2),
        ell_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, disk=disk)

    return al.Tracer(
        galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.wrap.Planck15()
    )


@pytest.fixture(name="tracer_dict")
def make_tracer_dict():
    return {
        "type": "instance",
        "class_path": "autolens.lens.ray_tracing.Tracer",
        "arguments": {
            "run_time_dict": None,
            "galaxies": {
                "type": "list",
                "values": [
                    {
                        "type": "instance",
                        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                        "arguments": {
                            "redshift": 0.5,
                            "label": "cls267",
                            "mass": {
                                "type": "instance",
                                "class_path": "autogalaxy.profiles.mass.total.isothermal.Isothermal",
                                "arguments": {
                                    "einstein_radius": 1.6,
                                    "ell_comps": (0.1, 0.05),
                                    "centre": (0.0, 0.0),
                                },
                            },
                        },
                    },
                    {
                        "type": "instance",
                        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                        "arguments": {
                            "redshift": 1.0,
                            "label": "cls267",
                            "disk": {
                                "type": "instance",
                                "class_path": "autogalaxy.profiles.light.standard.exponential.Exponential",
                                "arguments": {
                                    "effective_radius": 0.5,
                                    "intensity": 0.05,
                                    "ell_comps": (0.05, 0.25),
                                    "centre": (0.3, 0.2),
                                },
                            },
                        },
                    },
                ],
            },
            "cosmology": {
                "type": "instance",
                "class_path": "autogalaxy.cosmology.wrap.Planck15",
                "arguments": {},
            },
        },
    }


def test__to_dict(tracer, tracer_dict):
    assert to_dict(tracer) == tracer_dict


def test__from_dict(tracer, tracer_dict):
    assert from_dict(tracer_dict) == tracer
