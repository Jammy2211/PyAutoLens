import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit.tools.pipeline
import autolens.pipeline.phase.phase_imaging
from autofit import conf
from autofit.exc import PipelineException
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior
from autofit.optimize import non_linear
from autolens import exc
from autolens.data import ccd
from autolens.data.array import grids, mask as msk
from autolens.data.array import scaled_array
from autolens.lens import lens_data as ld
from autolens.lens import lens_fit
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline.phase import phase as ph

class MockAnalysis(object):

    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]

    def fit(self, instance):
        return 1


class MockResults(object):
    def __init__(self, model_image=None, galaxy_images=(), constant=None, analysis=None, optimizer=None):
        self.model_image = model_image
        self.galaxy_images = galaxy_images
        self.constant = constant or mm.ModelInstance()
        self.variable = mm.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer


class MockResult:
    def __init__(self, constant, figure_of_merit, variable=None):
        self.constant = constant
        self.figure_of_merit = figure_of_merit
        self.variable = variable
        self.previous_variable = variable
        self.gaussian_tuples = None


class MockNLO(non_linear.NonLinearOptimizer):

    def fit(self, analysis):

        class Fitness(object):

            def __init__(self, instance_from_physical_vector):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector

            def __call__(self, vector):

                instance = self.instance_from_physical_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = MockResult(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector)
        fitness_function(self.variable.prior_count * [0.8])

        return fitness_function.result