"""
Regression tests for the PyAutoLens#532 `Tracer` guard (B4).

Built from @rhayes777's own snippet in the issue body. Asserts the *failure*: that a
bad `galaxies` input is rejected at construction with a message naming `galaxies`,
rather than constructed and surfaced later as
`AttributeError: 'str' object has no attribute 'redshift'` — an error naming nothing
the caller passed.

The negative-redshift half of #532 is guarded in PyAutoGalaxy, where `Galaxy` and its
`redshift` assignment actually live (`al.Galaxy` IS `ag.Galaxy`), and is tested there.

`z_lens > z_source` is phase 4 of the audit and explicitly NOT implemented — it is
held pending the reporter's answer. The control at the bottom pins today's permissive
behaviour so phase 4 cannot regress it silently.
"""

import numpy as np
import pytest

import autofit as af
import autolens as al


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
# Phase 4 guard-rail — z_lens > z_source must NOT raise
# ======================================================================================


def test__control__lens_redshift_above_source_redshift_still_constructs_and_evaluates(
    grid,
):
    """
    PHASE 4 GUARD-RAIL — deliberately pinning today's permissive behaviour.

    Multi-plane lensing genuinely supports geometries that look inverted under
    two-plane naming, so this must not raise. Whether it should even *warn* is the
    open question put to @rhayes777 on #532. This test exists so phase 4 cannot
    quietly turn it into an error.
    """
    lens = al.Galaxy(redshift=1.0, mass=al.mp.IsothermalSph(einstein_radius=1.0))
    source = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    image = al.Tracer(galaxies=[lens, source]).image_2d_from(grid=grid)

    assert np.isfinite(np.asarray(image)).all()
    assert np.asarray(image).sum() > 0.0
