from autolens import conf
from autolens import exc
import math
import os
import pymultinest
import scipy.optimize
import numpy as np
from autolens.imaging import hyper_image
from autolens import conf
from autolens.autofit import model_mapper as mm
import logging
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__name__)

SIMPLEX_TUPLE_WIDTH = 0.1


class Result(object):

    def __init__(self, constant, likelihood, variable=None):
        """
        The result of an optimization.

        Parameters
        ----------
        constant: mm.ModelInstance
            An instance object comprising the class instances that gave the optimal fit
        likelihood: float
            A value indicating the likelihood given by the optimal fit
        variable: mm.ModelMapper
            An object comprising priors determined by this stage of the lensing
        """
        self.constant = constant
        self.likelihood = likelihood
        self.variable = variable

    def __str__(self):
        return "Analysis Result:\n{}".format(
            "\n".join(["{}: {}".format(key, value) for key, value in self.__dict__.items()]))


class NonLinearOptimizer(object):

    def __init__(self, include_hyper_image=False, model_mapper=None, name=None, label_config=None, **classes):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer nlo, which are standardized across all \
        non-linear optimizers.

        Parameters
        ------------
        path : str
            The path where the non-linear lensing nlo are stored.
        obj_name : str
            Unique identifier of the data_vector being analysed (e.g. the analysis_path of the data_vector set)
        """

        self.named_config = conf.instance.non_linear

        if name is None:
            name = ""
        self.path = "{}/{}".format(conf.instance.output_path, name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.chains_path = "{}/{}".format(self.path, 'chains')
        if not os.path.exists(self.chains_path):
            os.makedirs(self.chains_path)

        self.variable = model_mapper or mm.ModelMapper()
        self.constant = mm.ModelInstance()

        self.label_config = label_config or conf.instance.label

        self.file_param_names = "{}/{}".format(self.chains_path, 'mn.paramnames')
        self.file_model_info = "{}/{}".format(self.path, 'model.info')

        # If the include_hyper_image flag is set to True make this an additional prior model
        if include_hyper_image:
            self.hyper_image = mm.PriorModel(hyper_image.HyperImage, config=self.variable.config)

    def config(self, attribute_name, attribute_type=str):
        """
        Get a config field from this optimizer's section in non_linear.ini by a key and value type.

        Parameters
        ----------
        attribute_name: str
            The analysis_path of the field
        attribute_type: type
            The type of the value

        Returns
        -------
        attribute
            An attribute for the key with the specified type.
        """
        return self.named_config.get(self.__class__.__name__, attribute_name, attribute_type)

    def save_model_info(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)  # Create results folder if doesnt exist

        self.create_paramnames_file()
        if not os.path.isfile(self.file_model_info):
            with open(self.file_model_info, 'w') as file:
                file.write(self.variable.model_info)
            file.close()

    def fit(self, analysis):
        raise NotImplementedError("Fitness function must be overridden by non linear optimizers")

    @property
    def param_names(self):
        """The param_names vector is a list each parameter's analysis_path, and is used for *GetDist* visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are \
        properties of each model class."""

        paramnames_names = []

        prior_prior_model_name_dict = self.variable.prior_prior_model_name_dict

        for prior_name, prior in self.variable.prior_tuples_ordered_by_id:
            paramnames_names.append(prior_prior_model_name_dict[prior] + '_' + prior_name)

        return paramnames_names

    @property
    def constant_names(self):
        constant_names = []

        constant_prior_model_name_dict = self.variable.constant_prior_model_name_dict

        for constant_name, constant in self.variable.constant_tuples_ordered_by_id:
            constant_names.append(constant_prior_model_name_dict[constant] + '_' + constant_name)

        return constant_names

    @property
    def param_labels(self):
        """The param_names vector is a list each parameter's analysis_path, and is used for *GetDist* visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""

        paramnames_labels = []
        prior_class_dict = self.variable.prior_class_dict
        prior_prior_model_dict = self.variable.prior_prior_model_dict

        for prior_name, prior in self.variable.prior_tuples_ordered_by_id:
            param_string = self.label_config.label(prior_name)
            prior_model = prior_prior_model_dict[prior]
            cls = prior_class_dict[prior]
            cls_string = "{}{}".format(self.label_config.subscript(cls), prior_model.component_number + 1)
            param_label = "{}_{{\\mathrm{{{}}}}}".format(param_string, cls_string)
            paramnames_labels.append(param_label)

        return paramnames_labels

    def create_paramnames_file(self):
        """The param_names file lists every parameter's analysis_path and Latex tag, and is used for *GetDist*
        visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""
        paramnames_names = self.param_names
        paramnames_labels = self.param_labels
        with open(self.file_param_names, 'w') as paramnames:
            for i in range(self.variable.total_parameters):
                line = paramnames_names[i]
                line += ' ' * (50 - len(line)) + paramnames_labels[i]
                paramnames.write(line + '\n')


class DownhillSimplex(NonLinearOptimizer):

    def __init__(self, include_hyper_image=False, model_mapper=None,
                 fmin=scipy.optimize.fmin, name=None, label_config=None):
        super(DownhillSimplex, self).__init__(include_hyper_image=include_hyper_image,
                                              model_mapper=model_mapper, name=name, label_config=label_config)

        self.xtol = self.config("xtol", float)
        self.ftol = self.config("ftol", float)
        self.maxiter = self.config("maxiter", int)
        self.maxfun = self.config("maxfun", int)

        self.full_output = self.config("full_output", int)
        self.disp = self.config("disp", int)
        self.retall = self.config("retall", int)

        self.fmin = fmin

        logger.debug("Creating DownhillSimplex NLO")

    def fit(self, analysis):
        initial_vector = self.variable.physical_values_from_prior_medians

        class Fitness(object):
            def __init__(self, instance_from_physical_vector, constant):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.constant = constant

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)

                instance += self.constant

                likelihood = analysis.fit(instance)
                self.result = Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)

        logger.info("Running DownhillSimplex...")
        output = self.fmin(fitness_function, x0=initial_vector)
        logger.info("DownhillSimplex complete")
        res = fitness_function.result

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res.variable = self.variable.mapper_from_gaussian_means(output)

        return res


class MultiNest(NonLinearOptimizer):

    def __init__(self, include_hyper_image=False, model_mapper=None, sigma_limit=3, run=pymultinest.run, name=None,
                 label_config=None):
        """Class to setup and run a MultiNest lensing and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances that \
        are passed to each iteration of MultiNest.

        Parameters
        ------------
        path : str
            The path where the non_linear nlo are stored.
        """

        super(MultiNest, self).__init__(include_hyper_image=include_hyper_image, model_mapper=model_mapper, name=name,
                                        label_config=label_config)

        self.file_summary = "{}/{}".format(self.chains_path, 'mnsummary.txt')
        self.file_weighted_samples = "{}/{}".format(self.chains_path, 'mn.txt')
        self.file_results = "{}/{}".format(self.path, 'mn.results')
        self._weighted_sample_model = None
        self.sigma_limit = sigma_limit

        self.importance_nested_sampling = self.config('importance_nested_sampling', bool)
        self.multimodal = self.config('multimodal', bool)
        self.const_efficiency_mode = self.config('const_efficiency_mode', bool)
        self.n_live_points = self.config('n_live_points', int)
        self.evidence_tolerance = self.config('evidence_tolerance', float)
        self.sampling_efficiency = self.config('sampling_efficiency', float)
        self.n_iter_before_update = self.config('n_iter_before_update', int)
        self.null_log_evidence = self.config('null_log_evidence', float)
        self.max_modes = self.config('max_modes', int)
        self.mode_tolerance = self.config('mode_tolerance', float)
        self.outputfiles_basename = self.config('outputfiles_basename', str)
        self.seed = self.config('seed', int)
        self.verbose = self.config('verbose', bool)
        self.resume = self.config('resume', bool)
        self.context = self.config('context', int)
        self.write_output = self.config('write_output', bool)
        self.log_zero = self.config('log_zero', float)
        self.max_iter = self.config('max_iter', int)
        self.init_MPI = self.config('init_MPI', bool)
        self.run = run

        logger.debug("Creating MultiNest NLO")

    @property
    def pdf(self):
        import getdist
        return getdist.mcsamples.loadMCSamples(self.chains_path + '/mn')

    def fit(self, analysis):

        if len(self.path) > 77:
            raise exc.MultiNestException(
                'The path to the MultiNest results is longer than 77 characters (={}). Unfortunately, PyMultiNest '
                'cannot use a path longer than this. Set your results path to something with fewer characters to '
                'fix.'.format(len(self.path)))

        self.save_model_info()

        class Fitness(object):

            def __init__(self, instance_from_physical_vector, _constant, output_results):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.constant = _constant
                self.max_likelihood = -np.inf
                self.output_results = output_results
                self.accepted_samples = 0

            def __call__(self, cube, ndim, nparams, lnew):

                instance = self.instance_from_physical_vector(cube)
                instance += self.constant

                try:
                    likelihood = analysis.fit(instance)
                except exc.InversionException or exc.RayTracingException:
                    likelihood = -np.inf

                # TODO: Use multinest to provide best model

                if likelihood > self.max_likelihood:

                    # TODO : make the 10 below a config file param e.g. output_results_every_accepted_samples

                    self.accepted_samples += 1

                    if self.accepted_samples == 10:
                        self.accepted_samples = 0
                        self.output_results(during_analysis=True)

                    self.max_likelihood = likelihood
                    self.result = Result(instance, likelihood)

                return likelihood

        # noinspection PyUnusedLocal
        def prior(cube, ndim, nparams):

            phys_cube = self.variable.physical_vector_from_hypercube_vector(hypercube_vector=cube)

            for i in range(self.variable.total_parameters):
                cube[i] = phys_cube[i]

            return cube

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant, self.output_results)

        logger.info("Running MultiNest...")
        self.run(fitness_function.__call__, prior, self.variable.total_parameters,
                 outputfiles_basename="{}/mn".format(self.chains_path), n_live_points=self.n_live_points,
                 const_efficiency_mode=self.const_efficiency_mode,
                 importance_nested_sampling=self.importance_nested_sampling,
                 evidence_tolerance=self.evidence_tolerance, sampling_efficiency=self.sampling_efficiency,
                 null_log_evidence=self.null_log_evidence, n_iter_before_update=self.n_iter_before_update,
                 multimodal=self.multimodal, max_modes=self.max_modes, mode_tolerance=self.mode_tolerance,
                 seed=self.seed,
                 verbose=self.verbose, resume=self.resume, context=self.context, write_output=self.write_output,
                 log_zero=self.log_zero, max_iter=self.max_iter, init_MPI=self.init_MPI)
        logger.info("MultiNest complete")

        self.output_results(during_analysis=False)
        self.output_pdf_plots()

        constant = self.most_likely_instance_from_summary()
        constant += self.constant
        likelihood = self.max_likelihood_from_summary()
        variable = self.variable.mapper_from_gaussian_tuples(
            tuples=self.gaussian_priors_at_sigma_limit(self.sigma_limit))

        return Result(constant=constant, likelihood=likelihood, variable=variable)

    def open_summary_file(self):

        summary = open(self.file_summary)

        expected_parameters = (len(summary.readline()) - 113) / 112

        if expected_parameters != self.variable.total_parameters:
            raise exc.MultiNestException(
                'The file_summary file has a different number of parameters than the input model')

        return summary

    def read_vector_from_summary(self, number_entries, offset):

        summary = self.open_summary_file()

        summary.seek(0)
        summary.read(2 + offset * self.variable.total_parameters)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector

    def most_probable_from_summary(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        return self.read_vector_from_summary(number_entries=self.variable.total_parameters, offset=0)

    def most_likely_from_summary(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.read_vector_from_summary(number_entries=self.variable.total_parameters, offset=56)

    def max_likelihood_from_summary(self):
        return self.read_vector_from_summary(number_entries=2, offset=112)[0]

    def max_log_likelihood_from_summary(self):
        return self.read_vector_from_summary(number_entries=2, offset=112)[1]

    def most_probable_instance_from_summary(self):
        most_probable = self.most_probable_from_summary()
        return self.variable.instance_from_physical_vector(most_probable)

    def most_likely_instance_from_summary(self):
        most_likely = self.most_likely_from_summary()
        return self.variable.instance_from_physical_vector(most_likely)

    def gaussian_priors_at_sigma_limit(self, sigma_limit):
        """Compute the Gaussian Priors these results should be initialzed with in the next phase, by taking their \
        most probable values (e.g the means of their PDF) and computing the error at an input sigma_limit.

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.most_probable_from_summary()
        uppers = self.model_at_upper_sigma_limit(sigma_limit)
        lowers = self.model_at_lower_sigma_limit(sigma_limit)

        # noinspection PyArgumentList
        sigmas = list(map(lambda mean, upper, lower: max([upper - mean, mean - lower]), means, uppers, lowers))

        return list(map(lambda mean, sigma: (mean, sigma), means, sigmas))

    def model_at_sigma_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))
        densities_1d = list(map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names))
        return list(map(lambda p: p.getLimits(limit), densities_1d))

    def model_at_upper_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[1], self.model_at_sigma_limit(sigma_limit)))

    def model_at_lower_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[0], self.model_at_sigma_limit(sigma_limit)))

    def weighted_sample_instance_from_weighted_samples(self, index):
        """Setup a model instance of a weighted sample, including its weight and likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        model, weight, likelihood = self.weighted_sample_model_from_weighted_samples(index)

        self._weighted_sample_model = model

        return self.variable.instance_from_physical_vector(model), weight, likelihood

    def weighted_sample_model_from_weighted_samples(self, index):
        """From a weighted sample return the model, weight and likelihood hood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        return list(self.pdf.samples[index]), self.pdf.weights[index], -0.5 * self.pdf.loglikes[index]

    def output_pdf_plots(self):

        import getdist.plots
        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.general.get('output', 'plot_pdf_1d_params', bool)

        if plot_pdf_1d_params:

            for param_name in self.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(fname=self.path + '/pdfs/' + param_name + '_1D.png')
                plt.close()

        plot_pdf_triangle = conf.instance.general.get('output', 'plot_pdf_triangle', bool)

        if plot_pdf_triangle:

            try:
                pdf_plot.triangle_plot(roots=self.pdf)
                pdf_plot.export(fname=self.path + '/pdfs/Triangle.png')
                plt.close()
            except np.linalg.LinAlgError:
                pass

    def output_results(self, during_analysis=False):

        if os.path.isfile(self.file_summary):

            with open(self.file_results, 'w') as results:

                max_likelihood = self.max_likelihood_from_summary()

                results.write('Most likely model, Likelihood = ' + str(max_likelihood) + '\n')
                results.write('\n')

                most_likely = self.most_likely_from_summary()

                for i in range(self.variable.total_parameters):
                    line = self.param_names[i]
                    line += ' ' * (50 - len(line)) + str(most_likely[i])
                    results.write(line + '\n')

                if during_analysis is False:

                    most_probable = self.most_probable_from_summary()

                    lower_limit = self.model_at_lower_sigma_limit(sigma_limit=3.0)
                    upper_limit = self.model_at_upper_sigma_limit(sigma_limit=3.0)

                    results.write('\n')
                    results.write('Most probable model (3 sigma limits)' + '\n')
                    results.write('\n')

                    for i in range(self.variable.total_parameters):
                        line = self.param_names[i]
                        line += ' ' * (50 - len(line)) + str(most_probable[i]) + ' (' + str(lower_limit[i]) + ', ' + str(
                            upper_limit[i]) + ')'
                        results.write(line + '\n')

                    lower_limit = self.model_at_lower_sigma_limit(sigma_limit=1.0)
                    upper_limit = self.model_at_upper_sigma_limit(sigma_limit=1.0)

                    results.write('\n')
                    results.write('Most probable model (1 sigma limits)' + '\n')
                    results.write('\n')

                    for i in range(self.variable.total_parameters):
                        line = self.param_names[i]
                        line += ' ' * (50 - len(line)) + str(most_probable[i]) + ' (' + str(lower_limit[i]) + ', ' + str(
                            upper_limit[i]) + ')'
                        results.write(line + '\n')

                results.write('\n')
                results.write('Constants' + '\n')
                results.write('\n')

                constant_names = self.constant_names
                constants = self.variable.constant_tuples_ordered_by_id

                for i in range(self.variable.total_constants):
                    line = constant_names[i]
                    line += ' ' * (50 - len(line)) + str(constants[i][1].value)
