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

    return al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.wrap.Planck15
    )


@pytest.fixture(name="tracer_dict")
def make_tracer_dict():
    return {
        "type": "instance",
        "class_path": "autolens.lens.ray_tracing.Tracer",
        "arguments": {
            "run_time_dict": None,
            "planes": {
                "type": "list",
                "values": [
                    {
                        "type": "instance",
                        "class_path": "autogalaxy.plane.plane.Plane",
                        "arguments": {
                            "galaxies": {
                                "type": "list",
                                "values": [
                                    {
                                        "type": "instance",
                                        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                                        "arguments": {
                                            "redshift": 0.5,
                                            "mass": {
                                                "type": "instance",
                                                "class_path": "autogalaxy.profiles.mass.total.isothermal.Isothermal",
                                                "arguments": {
                                                    "ell_comps": (0.1, 0.05),
                                                    "centre": (0.0, 0.0),
                                                    "einstein_radius": 1.6,
                                                },
                                            },
                                        },
                                    }
                                ],
                            },
                            "run_time_dict": None,
                            "redshift": 0.5,
                        },
                    },
                    {
                        "type": "instance",
                        "class_path": "autogalaxy.plane.plane.Plane",
                        "arguments": {
                            "galaxies": {
                                "type": "list",
                                "values": [
                                    {
                                        "type": "instance",
                                        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                                        "arguments": {
                                            "redshift": 1.0,
                                            "disk": {
                                                "type": "instance",
                                                "class_path": "autogalaxy.profiles.light.standard.exponential.Exponential",
                                                "arguments": {
                                                    "ell_comps": (0.05, 0.25),
                                                    "centre": (0.3, 0.2),
                                                    "effective_radius": 0.5,
                                                    "intensity": 0.05,
                                                },
                                            },
                                        },
                                    }
                                ],
                            },
                            "run_time_dict": None,
                            "redshift": 1.0,
                        },
                    },
                ],
            },
            "cosmology": {
                "type": "type",
                "class_path": "autogalaxy.cosmology.wrap.Planck15",
            },
        },
    }


def test__to_dict(tracer, tracer_dict):
    assert to_dict(tracer) == tracer_dict


def test__from_dict(tracer, tracer_dict):
    assert from_dict(tracer_dict) == tracer
