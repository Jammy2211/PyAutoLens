from os import path

import numpy as np
import pytest

import autofit as af
from autofit.mapper.prior.prior import TuplePrior
import autolens as al
from autolens.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestExtensions:
    def test__extend_with_stochastic_phase__sets_up_model_correctly(self, mask_7x7):
        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5,
            light=al.lp.SphericalSersic(),
            mass=al.mp.SphericalIsothermal(),
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(),
            regularization=al.reg.AdaptiveBrightness(),
        )

        phase = al.PhaseImaging(
            search=mock.MockSearch(),
            galaxies=af.CollectionPriorModel(lens=al.GalaxyModel(redshift=0.5)),
        )

        phase_extended = phase.extend_with_stochastic_phase()

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(model.source.regularization.inner_coefficient, float)

        phase_extended = phase.extend_with_stochastic_phase(include_lens_light=True)

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, af.UniformPrior)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(model.source.regularization.inner_coefficient, float)

        phase_extended = phase.extend_with_stochastic_phase(include_pixelization=True)

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, af.UniformPrior)
        assert not isinstance(
            model.source.regularization.inner_coefficient, af.UniformPrior
        )

        phase_extended = phase.extend_with_stochastic_phase(include_regularization=True)

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(
            model.source.regularization.inner_coefficient, af.UniformPrior
        )

        phase = al.PhaseInterferometer(
            search=mock.MockSearch(),
            real_space_mask=mask_7x7,
            galaxies=af.CollectionPriorModel(lens=al.GalaxyModel(redshift=0.5)),
        )

        phase_extended = phase.extend_with_stochastic_phase()

        model = phase_extended.make_model(instance=galaxies)

        assert isinstance(model.lens.mass.centre, TuplePrior)
        assert isinstance(model.lens.light.intensity, float)
        assert isinstance(model.source.pixelization.pixels, int)
        assert isinstance(model.source.regularization.inner_coefficient, float)
