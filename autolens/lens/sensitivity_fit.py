from autolens import exc
from autolens.lens import lens_fit
from autolens.lens import ray_tracing

def fit_lens_data_with_sensitivity_tracers(lens_data, tracer_normal, tracer_sensitive):
    """Fit lens data with a normal tracer and sensitivity tracer, to determine our sensitivity to a selection of \ 
    galaxy components. This factory automatically determines the type of fit based on the properties of the galaxies \
    in the tracers.

    Parameters
    -----------
    lens_data : lens_data.LensData or lens_data.LensDataHyper
        The lens-images that is fitted.
    tracer_normal : ray_tracing.AbstractTracer
        A tracer whose galaxies have the same model components (e.g. light profiles, mass profiles) as the \
        lens data that we are fitting.
    tracer_sensitive : ray_tracing.AbstractTracerNonStack
        A tracer whose galaxies have the same model components (e.g. light profiles, mass profiles) as the \
        lens data that we are fitting, but also addition components (e.g. mass clumps) which we measure \
        how sensitive we are too.
    """

    if (tracer_normal.has_light_profile and tracer_sensitive.has_light_profile) and \
            (not tracer_normal.has_pixelization and not tracer_sensitive.has_pixelization):
        return SensitivityProfileFit(lens_data=lens_data, tracer_normal=tracer_normal,
                                     tracer_sensitive=tracer_sensitive)

    elif (not tracer_normal.has_light_profile and not tracer_sensitive.has_light_profile) and \
            (tracer_normal.has_pixelization and tracer_sensitive.has_pixelization):
        return SensitivityInversionFit(lens_data=lens_data, tracer_normal=tracer_normal,
                                     tracer_sensitive=tracer_sensitive)
    else:

        raise exc.FittingException('The sensitivity_fit routine did not call a SensitivityFit class - check the '
                                   'properties of the tracers')


class AbstractSensitivityFit(object):

    def __init__(self, tracer_normal, tracer_sensitive):

        self.tracer_normal = tracer_normal
        self.tracer_sensitive = tracer_sensitive


class SensitivityProfileFit(AbstractSensitivityFit):

    def __init__(self, lens_data, tracer_normal, tracer_sensitive):
        """Evaluate the sensitivity of a profile fit to a specific component of a lens model and tracer. This is \
        performed by evaluating the likelihood of a fit to an image using two tracers:

        1) A 'normal tracer', which uses the same lens model as a the simulated lens data. This gives a baseline \
           value of the likelihood we can expect when we fit the model to itself.

        2) A 'sensitive tracer', which uses the same lens model as the simulated lens data, but also includes the \
           additional model components (e.g. a mass clump 'subhalo') which we are testing our sensitivity to.

        The difference in likelihood of these two fits informs us of how sensitive we are to the component in the \
        second tracer. For example, if the difference in likelihood is neglible, it means the model component had no \
        impact on our fit, meaning we are not sensitive to its properties.

        Parameters
        ----------
        lens_data: lens_data.LensData
            A simulated lens data which is used to determine our sensitiivity to specific model components.
        tracer_normal : ray_tracing.AbstractTracer
            A tracer whose galaxies have the same model components (e.g. light profiles, mass profiles) as the \
            lens data that we are fitting.
       tracer_sensitive : ray_tracing.AbstractTracerNonStack
            A tracer whose galaxies have the same model components (e.g. light profiles, mass profiles) as the \
            lens data that we are fitting, but also addition components (e.g. mass clumps) which we measure \
            how sensitive we are too.
        """
        AbstractSensitivityFit.__init__(self=self, tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)
        self.fit_normal = lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer_normal)
        self.fit_sensitive = lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer_sensitive)

    @property
    def figure_of_merit(self):
        return self.fit_sensitive.likelihood - self.fit_normal.likelihood


class SensitivityInversionFit(AbstractSensitivityFit):

    def __init__(self, lens_data, tracer_normal, tracer_sensitive):
        """Evaluate the sensitivity of an invesion fit to a specific component of a lens model and tracer. This is \
        performed by evaluating the likelihood of a fit to an image using two tracers:

        1) A 'normal tracer', which uses the same lens model as a the simulated lens data. This gives a baseline \
           value of the likelihood we can expect when we fit the model to itself.

        2) A 'sensitive tracer', which uses the same lens model as the simulated lens data, but also includes the \
           additional model components (e.g. a mass clump 'subhalo') which we are testing our sensitivity to.

        The difference in likelihood of these two fits informs us of how sensitive we are to the component in the \
        second tracer. For example, if the difference in likelihood is neglible, it means the model component had no \
        impact on our fit, meaning we are not sensitive to its properties.

        Parameters
        ----------
        lens_data: lens_data.LensData
            A simulated lens data which is used to determine our sensitiivity to specific model components.
        tracer_normal : ray_tracing.AbstractTracer
            A tracer whose galaxies have the same model components (e.g. light profiles, mass profiles) as the \
            lens data that we are fitting.
       tracer_sensitive : ray_tracing.AbstractTracerNonStack
            A tracer whose galaxies have the same model components (e.g. light profiles, mass profiles) as the \
            lens data that we are fitting, but also addition components (e.g. mass clumps) which we measure \
            how sensitive we are too.
        """
        AbstractSensitivityFit.__init__(self=self, tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)
        self.fit_normal = lens_fit.LensInversionFit(lens_data=lens_data, tracer=tracer_normal)
        self.fit_sensitive = lens_fit.LensInversionFit(lens_data=lens_data, tracer=tracer_sensitive)

    @property
    def figure_of_merit(self):
        return self.fit_sensitive.likelihood - self.fit_normal.likelihood