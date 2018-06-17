import getdist
from auto_lens import exc
import math
import os
import pymultinest
import scipy.optimize

from auto_lens.analysis import model_mapper as mm

default_path = '{}/../output/'.format(os.path.dirname(os.path.realpath(__file__)))


def generate_parameter_latex(parameters, subscript=''):
    """Generate a latex label for a non-linear search parameter.

    This is used for the param names file and outputting the files of a run to a latex table.

    Parameters
    ----------
    parameters : [str]
        The parameter names to be converted to latex.
    subscript : str
        The subscript of the latex entry, often giving the parameter type (e.g. light or dark matter) or numerical \
        number of the component of the model_mapper.

    """

    latex = []

    if subscript == '':
        for param in parameters:
            latex.append('$' + param + '$')
    else:
        for param in parameters:
            latex.append('$' + param + r'_{\mathrm{' + subscript + '}}$')

    return latex


class NonLinearOptimizer(object):

    def __init__(self, config_path=None, path=default_path, check_model=True):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer files, which are standardized across all \
        non-linear optimizers.

        Parameters
        ------------
        path : str
            The path where the non-linear analysis files are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        check_model : bool
            Check whether the model.info file corresponds to the model_mapper passed in.
        """

        self.path = path
        self.check_model = check_model
        config = mm.Config(
            "{}/../config".format(os.path.dirname(os.path.realpath(__file__))) if config_path is None else config_path)
        self.model_mapper = mm.ModelMapper(config)

        self.file_param_names = self.path + 'multinest.paramnames'
        self.file_model_info = self.path + 'model.info'

    def save_model_info(self):
        print("making dir {}".format(self.path))
        resume = os.path.exists(self.path)  # resume True if results path already exists

        if not resume:
            os.makedirs(self.path)  # Create results folder if doesnt exist
            self.create_param_names()
            self.model_mapper.output_model_info(self.file_model_info)

        elif self.check_model:
            self.model_mapper.check_model_info(self.file_model_info)

    def run(self, fitness_function):
        raise NotImplementedError("Fitness function must be overridden by non linear optimizers")

    def compute_gaussian_priors(self, sigma_limit):
        raise NotImplementedError("Gaussian priors function must be overridden by non linear optimizers")

    def create_param_names(self):
        """The param_names file lists every parameter's name and Latex tag, and is used for *GetDist* visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are \
        properties of each model class."""
        param_names = open(self.file_param_names, 'w')

        for prior_name, prior_model in self.model_mapper.prior_models:

            param_labels = prior_model.cls.parameter_labels.__get__(prior_model.cls)
            component_number = prior_model.cls().component_number
            subscript = prior_model.cls.subscript.__get__(prior_model.cls) + str(component_number + 1)

            param_labels = generate_parameter_latex(param_labels, subscript)

            for param_no, param in enumerate(self.model_mapper.class_priors_dict[prior_name]):
                line = prior_name + '_' + param[0]
                line += ' ' * (40 - len(line)) + param_labels[param_no]

                param_names.write(line + '\n')

        param_names.close()


# TODO : Integratioin tests for this?? Hard to test as a unit test.
# TODO : Need to think how this interfaces with Prior intialization.

class DownhillSimplex(NonLinearOptimizer):

    def __init__(self, config_path, path=default_path):

        super(DownhillSimplex, self).__init__(config_path, path, False)

    def run(self, fitness_function):

        initlal_model = self.model_mapper.physical_values_from_prior_medians()
        return scipy.optimize.fmin(fitness_function, x0=initlal_model)


class MultiNest(NonLinearOptimizer):

    def __init__(self, config_path=None, path=default_path, check_model=True):
        """Class to setup and run a MultiNest analysis and output the MultInest files.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances that \
        are passed to each iteration of MultiNest.

        Parameters
        ------------
        path : str
            The path where the non_linear files are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        """

        super(MultiNest, self).__init__(config_path, path, check_model)

        self.file_summary = self.path + 'summary.txt'
        self.file_weighted_samples = self.path + 'multinest.txt'

    @property
    def pdf(self):
        return getdist.mcsamples.loadMCSamples(self.file_weighted_samples)

    def run(self, fitness_function):
        self.save_model_info()

        # noinspection PyUnusedLocal
        def prior(cube, ndim, nparams):
            return map(lambda p, c: p(c), self.model_mapper.total_parameters, cube)

        # TODO: is this output path correct? No - I have changed it to just the path.
        pymultinest.run(fitness_function, prior, self.model_mapper.total_parameters,
                        outputfiles_basename=self.path)

    def open_summary_file(self):

        summary = open(self.file_summary)

        expected_parameters = (len(summary.readline()) - 57) / 56
        if expected_parameters != self.model_mapper.total_parameters:
            raise exc.MultiNestException(
                'The file_summary file has a different number of parameters than the input model')

        return summary

    def read_vector_from_summary(self, number_entries, offset):

        summary = self.open_summary_file()

        summary.seek(0)
        summary.read(2 + offset * self.model_mapper.total_parameters)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector

    def compute_most_probable(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which files from a \
        multinest analysis.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        Parameters
        -----------
        filename : str
            The files and file name of the file_summary file.
        total_parameters : int
            The total number of parameters of the model.
        offset : int
            The file_summary file stores the most likely model in the first half of columns and the most probable model in
            the second half. The offset is used to start the parsing at the appropriate column.
        """
        return self.read_vector_from_summary(number_entries=self.model_mapper.total_parameters, offset=0)

    def compute_most_likely(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which files from a \
        multinest analysis.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.read_vector_from_summary(number_entries=self.model_mapper.total_parameters, offset=28)

    def compute_max_likelihood(self):
        return self.read_vector_from_summary(number_entries=2, offset=56)[0]

    def compute_max_log_likelihood(self):
        return self.read_vector_from_summary(number_entries=2, offset=56)[1]

    def create_most_probable_model_instance(self):
        most_probable = self.compute_most_probable()
        return self.model_mapper.from_physical_vector(most_probable)

    def create_most_likely_model_instance(self):
        most_likely = self.compute_most_likely()
        return self.model_mapper.from_physical_vector(most_likely)

    def compute_gaussian_priors(self, sigma_limit):
        """Compute the Gaussian Priors these results should be initialzed with in the next phase, by taking their \
        most probable values (e.g the means of their PDF) and computing the error at an input sigma_limit.

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.compute_most_probable()
        uppers = self.compute_model_at_upper_limit(sigma_limit)
        lowers = self.compute_model_at_lower_limit(sigma_limit)

        sigmas = list(map(lambda mean, upper, lower: max([upper - mean, mean - lower]), means, uppers, lowers))

        return list(map(lambda mean, sigma: (mean, sigma), means, sigmas))

    def compute_model_at_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))
        densities_1d = list(map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names))
        return list(map(lambda p: p.getLimits(limit), densities_1d))

    def compute_model_at_upper_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest files.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[1], self.compute_model_at_limit(sigma_limit)))

    def compute_model_at_lower_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest files.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[0], self.compute_model_at_limit(sigma_limit)))

    def create_weighted_sample_model_instance(self, index):
        """Setup a model instance of a weighted sample, including its weight and likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        model, weight, likelihood = self.compute_weighted_sample_model(index)

        self._weighted_sample_model = model

        return self.model_mapper.from_physical_vector(model), weight, likelihood

    def compute_weighted_sample_model(self, index):
        """From a weighted sample return the model, weight and likelihood hood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        return list(self.pdf.samples[index]), self.pdf.weights[index], -0.5 * self.pdf.loglikes[index]

    # TODO : untested and unfinished, remains to be seen if we'll need this code.

    def reorder_summary_file(self, new_order):
        most_probable = self.compute_most_probable()
        most_likely = self.compute_most_likely()
        likelihood = self.compute_max_likelihood()[0]
        log_likelihood = self.compute_max_likelihood()[1]

        most_probable = list(map(lambda param: ('%18.18E' % param).rjust(28), most_probable))
        most_probable = ''.join(map(str, most_probable))
        most_likely = list(map(lambda param: ('%18.18E' % param).rjust(28), most_likely))
        most_likely = ''.join(map(str, most_likely))
        likelihood = ('%18.18E' % 0.0).rjust(28)
        log_likelihood = ('%18.18E' % 0.0).rjust(28)

        new_summary_file = open(self.path + 'summary_new.txt', 'w')
        new_summary_file.write(most_probable + most_likely + likelihood + log_likelihood)
        new_summary_file.close()
