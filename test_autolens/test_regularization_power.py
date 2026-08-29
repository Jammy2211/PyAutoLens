"""
Model-composition gate for the ``*Power`` regularization siblings added in PyAutoArray, through the
``al.reg`` namespace.

Mirrors ``test_autogalaxy/test_regularization_power.py``; the prior entries this exercises live in
``test_autolens/config/priors/regularization.yaml``.
"""

import autofit as af
import autolens as al


def test__power_classes_are_re_exported():
    assert al.reg.AdaptPower is not al.reg.Adapt
    assert al.reg.AdaptSplitPower is not al.reg.AdaptSplit
    assert al.reg.AdaptSplitZerothPower is not al.reg.AdaptSplitZeroth
    assert al.reg.MaternAdaptPowerKernel is not al.reg.MaternAdaptKernel


def test__model_composition__power_is_a_constant_and_is_not_sampled():
    model = af.Model(al.reg.AdaptSplitPower)

    assert set(prior_tuple[0] for prior_tuple in model.prior_tuples) == {
        "inner_coefficient",
        "outer_coefficient",
        "signal_scale",
    }
    assert model.instance_from_prior_medians().power == 1.0


def test__model_identifier__differs_from_the_legacy_class():
    assert af.Model(al.reg.AdaptSplit).identifier != af.Model(al.reg.AdaptSplitPower).identifier
