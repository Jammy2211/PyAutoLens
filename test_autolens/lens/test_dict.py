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
        galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.wrap.Planck15
    )


@pytest.fixture(name="tracer_dict")
def make_tracer_dict():
    return {
        "arguments": {
            "cosmology": {
                "class_path": "autogalaxy.cosmology.wrap.Planck15",
                "type": "type",
            },
            "planes": {
                "type": "list",
                "values": [
                    {
                        "arguments": {
                            "galaxies": {
                                "type": "list",
                                "values": [
                                    {
                                        "arguments": {
                                            "label": "cls267",
                                            "mass": {
                                                "arguments": {
                                                    "centre": (0.0, 0.0),
                                                    "einstein_radius": 1.6,
                                                    "ell_comps": (0.1, 0.05),
                                                },
                                                "class_path": "autogalaxy.profiles.mass.total.isothermal.Isothermal",
                                                "type": "instance",
                                            },
                                            "redshift": 0.5,
                                        },
                                        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                                        "type": "instance",
                                    }
                                ],
                            },
                            "redshift": 0.5,
                            "run_time_dict": None,
                        },
                        "class_path": "autogalaxy.plane.plane.Plane",
                        "type": "instance",
                    },
                    {
                        "arguments": {
                            "galaxies": {
                                "type": "list",
                                "values": [
                                    {
                                        "arguments": {
                                            "disk": {
                                                "arguments": {
                                                    "centre": (0.3, 0.2),
                                                    "effective_radius": 0.5,
                                                    "ell_comps": (0.05, 0.25),
                                                    "intensity": 0.05,
                                                },
                                                "class_path": "autogalaxy.profiles.light.standard.exponential.Exponential",
                                                "type": "instance",
                                            },
                                            "label": "cls267",
                                            "redshift": 1.0,
                                        },
                                        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                                        "type": "instance",
                                    }
                                ],
                            },
                            "redshift": 1.0,
                            "run_time_dict": None,
                        },
                        "class_path": "autogalaxy.plane.plane.Plane",
                        "type": "instance",
                    },
                ],
            },
            "run_time_dict": None,
        },
        "class_path": "autolens.lens.ray_tracing.Tracer",
        "type": "instance",
    }


def test__to_dict(tracer, tracer_dict):
    assert to_dict(tracer) == tracer_dict


def test__from_dict(tracer, tracer_dict):
    assert from_dict(tracer_dict) == tracer
