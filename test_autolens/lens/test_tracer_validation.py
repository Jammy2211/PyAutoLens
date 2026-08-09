"""
Regression tests for the PyAutoLens#532 `Tracer` guard (B4).

Built from @rhayes777's own snippet in the issue body. Asserts the *failure*: that a
bad `galaxies` input is rejected at construction with a message naming `galaxies`,
rather than constructed and surfaced later as
`AttributeError: 'str' object has no attribute 'redshift'` — an error naming nothing
the caller passed.

The negative-redshift half of #532 is guarded in PyAutoGalaxy, where `Galaxy` and its
`redshift` assignment actually live (`al.Galaxy` IS `ag.Galaxy`), and is tested there.

`z_lens > z_source` (phase 4) is implemented at the bottom of this module as a
*warning* under its own filterable category — never an error, because multi-plane
lensing genuinely supports geometries that look inverted under two-plane naming.
"""

import warnings

import numpy as np
import pytest

import autofit as af
import autolens as al
from autolens.lens.tracer import MultiPlaneRedshiftWarning


@pytest.fixture(name="grid")
def make_grid():
    return al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.1)


@pytest.fixture(name="lens_galaxy")
def make_lens_galaxy():
    return al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))


@pytest.fixture(name="source_galaxy")
def make_source_galaxy():
    return al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))


# ======================================================================================
# B4 — galaxies must be an iterable of Galaxy
# ======================================================================================


def test__b4__a_string_is_rejected_even_though_a_string_is_iterable():
    """
    The trap in this finding: `isinstance(x, Iterable)` passes for a string, so
    iterability alone does not catch the reported case. The element type is what
    matters.
    """
    with pytest.raises(TypeError, match="galaxies"):
        al.Tracer(galaxies="not a list")


@pytest.mark.parametrize("galaxies", [42, None, {"a": 1}, 1.5, b"bytes"])
def test__b4__non_iterable_and_wrong_container_inputs_are_rejected(galaxies):
    """The reporter named the string case; these were all accepted too."""
    with pytest.raises(TypeError, match="galaxies"):
        al.Tracer(galaxies=galaxies)


def test__b4__a_list_containing_a_non_galaxy_is_rejected_and_names_the_index():
    with pytest.raises(TypeError, match="index 1"):
        al.Tracer(galaxies=[al.Galaxy(redshift=0.5), "not a galaxy"])


def test__b4__the_message_names_the_type_that_was_passed():
    with pytest.raises(TypeError) as error:
        al.Tracer(galaxies="not a list")

    assert "str" in str(error.value)
    assert "galaxies" in str(error.value)


# ======================================================================================
# Controls — valid inputs must keep working
# ======================================================================================


def test__control__a_normal_list_of_galaxies_builds_and_evaluates(
    grid, lens_galaxy, source_galaxy
):
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    image = tracer.image_2d_from(grid=grid)

    assert np.isfinite(np.asarray(image)).all()
    assert np.asarray(image).sum() > 0.0


def test__control__a_tuple_of_galaxies_is_accepted(lens_galaxy, source_galaxy):
    assert al.Tracer(galaxies=(lens_galaxy, source_galaxy)) is not None


def test__control__an_empty_list_is_accepted(lens_galaxy):
    """An empty tracer is degenerate but legal, and is used in tests and chaining."""
    assert al.Tracer(galaxies=[]) is not None


def test__control__a_model_instance_of_galaxies_is_accepted(lens_galaxy, source_galaxy):
    """PyAutoFit hands the tracer an `af.ModelInstance` during a model fit."""
    instance = af.ModelInstance()
    instance.lens = lens_galaxy
    instance.source = source_galaxy

    assert al.Tracer(galaxies=instance) is not None


# ======================================================================================
# Phase 4 — z_lens > z_source WARNS, and must never raise
# ======================================================================================
#
# Resolved 2026-08-09. @rhayes777 was asked on #532 whether a warning here would be
# noise in a real multi-plane setup and did not answer; the campaign was closed with
# the warning implemented behind its own filterable category, so a user for whom it
# IS noise can silence it with one filter rather than living with it.
#
# The rule is deliberately narrow — it fires only when NO light lies behind ANY mass,
# i.e. when nothing in the tracer can be lensed at all. Genuine multi-plane systems,
# where some light is lensed and some is not, stay quiet.


def _mass_galaxy(redshift):
    return al.Galaxy(
        redshift=redshift, mass=al.mp.IsothermalSph(einstein_radius=1.0)
    )


def _light_galaxy(redshift):
    return al.Galaxy(redshift=redshift, bulge=al.lp.Sersic(intensity=1.0))


def test__phase4__inverted_redshifts_warn_but_still_construct_and_evaluate(grid):
    """
    The reported case. It must WARN — and must still produce a finite image, because
    multi-plane genuinely supports geometries that look inverted under two-plane
    naming. A warning, never an error.
    """
    galaxies = [_mass_galaxy(1.0), _light_galaxy(0.5)]

    with pytest.warns(MultiPlaneRedshiftWarning, match="lies behind"):
        tracer = al.Tracer(galaxies=galaxies)

    image = tracer.image_2d_from(grid=grid)

    assert np.isfinite(np.asarray(image)).all()
    assert np.asarray(image).sum() > 0.0


def test__phase4__the_warning_names_both_sets_of_redshifts():
    with pytest.warns(MultiPlaneRedshiftWarning) as record:
        al.Tracer(galaxies=[_mass_galaxy(1.0), _light_galaxy(0.5)])

    message = str(record[0].message)

    assert "0.5" in message
    assert "1.0" in message


def test__phase4__the_warning_can_be_silenced_by_its_own_category():
    """
    The whole point of a dedicated category: a user for whom this is noise silences
    exactly this, and nothing else.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=MultiPlaneRedshiftWarning)

        al.Tracer(galaxies=[_mass_galaxy(1.0), _light_galaxy(0.5)])

    assert [w for w in caught if issubclass(w.category, MultiPlaneRedshiftWarning)] == []


@pytest.mark.parametrize(
    "label,galaxies_fn",
    [
        ("normal lens/source ordering", lambda: [_mass_galaxy(0.5), _light_galaxy(1.0)]),
        (
            "lens with its own light, plus a source behind it",
            lambda: [
                al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.IsothermalSph(einstein_radius=1.0),
                    bulge=al.lp.Sersic(intensity=1.0),
                ),
                _light_galaxy(1.0),
            ],
        ),
        (
            "genuine multi-plane: mass at 1.0, light at 0.5 AND 1.5",
            lambda: [_mass_galaxy(1.0), _light_galaxy(0.5), _light_galaxy(1.5)],
        ),
        ("everything on one plane", lambda: [_mass_galaxy(0.5), _light_galaxy(0.5)]),
        ("mass only, no light to lens", lambda: [_mass_galaxy(1.0), _mass_galaxy(0.5)]),
        ("light only, no mass", lambda: [_light_galaxy(1.0), _light_galaxy(0.5)]),
        ("empty tracer", lambda: []),
    ],
)
def test__phase4__legitimate_configurations_stay_quiet(label, galaxies_fn):
    """
    The noise question, answered by construction. The genuine multi-plane case is the
    one that matters most: mass at z=1.0 with light at BOTH 0.5 and 1.5 has some of
    its light lensed, so it is a real configuration and must not be flagged.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        al.Tracer(galaxies=galaxies_fn())

    assert [
        w for w in caught if issubclass(w.category, MultiPlaneRedshiftWarning)
    ] == [], label


@pytest.mark.parametrize(
    "label,source_galaxy_fn",
    [
        (
            "pixelized source — carries NO LightProfile but is still light to lens",
            lambda: al.Galaxy(
                redshift=1.0,
                pixelization=al.Pixelization(
                    mesh=al.mesh.RectangularUniform(shape=(3, 3)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            ),
        ),
        (
            "point source — carries neither light profile nor pixelization",
            lambda: al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.0, 0.0))),
        ),
    ],
)
def test__phase4__lensable_sources_without_a_light_profile_do_not_warn(
    label, source_galaxy_fn
):
    """
    Regression for the false-positive classes found while building this warning.

    Counting only `LightProfile` made this fire on every pixelized-source and
    point-source configuration in the suite — core PyAutoLens usage, and exactly the
    noise the reporter asked about. `LENSABLE_CLS` is the fix; this pins it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        al.Tracer(galaxies=[_mass_galaxy(0.5), source_galaxy_fn()])

    assert [
        w for w in caught if issubclass(w.category, MultiPlaneRedshiftWarning)
    ] == [], label


def test__phase4__an_empty_placeholder_galaxy_suppresses_the_warning():
    """
    A galaxy with nothing in it is a scaffold — a model being composed, or a source
    not yet filled in. Judging the geometry of a half-built system and warning about
    it is unhelpful, so the check stands down.
    """
    lens = al.Galaxy(
        redshift=0.5,
        light=al.lp.SersicSph(intensity=2.0),
        mass=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    empty_source = al.Galaxy(redshift=1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        al.Tracer(galaxies=[lens, empty_source])

    assert [
        w for w in caught if issubclass(w.category, MultiPlaneRedshiftWarning)
    ] == []


def test__phase4__a_non_concrete_redshift_skips_the_check_entirely():
    """
    Redshifts can be free model parameters (a traced subhalo redshift under jax.jit),
    so the check must skip rather than coerce a traced boolean.
    """

    class _TracerLikeRedshift:
        def __bool__(self):
            raise AssertionError("the warning compared a non-concrete redshift")

        def __lt__(self, other):
            return self

        def __gt__(self, other):
            return self

    galaxy = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    galaxy.redshift = _TracerLikeRedshift()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        al.Tracer(galaxies=[galaxy, _light_galaxy(0.5)])

    assert [
        w for w in caught if issubclass(w.category, MultiPlaneRedshiftWarning)
    ] == []
