import getdist
import sys
import os

class NonLinearDirectory(object):

    def __init__(self, path, obj_name, model_mapper):
        """Abstract base class for non-linear optimizers.

        Sets up the directory structure for their results.

        Parameters
        ------------
        path : str
            The path where the non-linear analysis results are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : CalibrationModel.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        """
        self.path = path
        self.obj_name = obj_name
        self.model_mapper = model_mapper
        self.total_parameters = len(self.model_mapper.priors_ordered_by_id)
        self.setup_results_path()
        self.resume = self.results_folder_exists()
        if self.resume == False:
            self.make_results_folder()

    def setup_results_path(self):
        """Use the model mapper to set up the results path of the non-linear analysis. This uses the classes that make \
        up the model mapper.
        """

        self.results_path = self.path + self.obj_name + '/'
        for prior_name, prior_model in self.model_mapper.prior_models:
            self.results_path += prior_model.cls.__name__
            self.results_path += '+'

        self.results_path = self.results_path[:-1] # remove last + symbol from path name
        self.results_path +='/'

    def results_folder_exists(self):
        return os.path.exists(self.results_path)

    def make_results_folder(self):
        os.makedirs(self.results_path)

    def generate_parameter_latex(self, parameters, subscript=''):
        """Generate a latex label for a non-linear search parameter.

        This is used for the paramnames file and outputting the results of a run to a latex table.

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


class MultiNestFiles(object):

    def __init__(self, obj_name, results_path):
        """Setup the path and file names of the MultiNestOptimizer files.

        Parameters
        ------------
        results_path : str
            The results path of the MultiNestOptimizer analysis.
        """
        self.param_names = results_path + obj_name + '.paramnames'
        self.model_info = results_path + 'model.info'
        self.weighted_samples = results_path + obj_name + '.txt'
        self.summary = results_path + 'summary.txt'


class MultiNestOptimizer(NonLinearDirectory):

    def __init__(self, path, obj_name, model_mapper):
        """Class to setup and run a MultiNestOptimizer analysis and output the results file.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances passed \
        to each iteration of MultiNestOptimizer.

        Parameters
        ------------
        path : str
            The path where the non_linear results are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : CalibrationModel.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        """

        super(MultiNestOptimizer, self).__init__(path, obj_name, model_mapper)

        self.files = MultiNestFiles(self.obj_name, self.results_path)

        if self.resume == False:
            self.make_param_names()
            self.output_model_info()
        elif self.resume == True:
            self.check_model_info()

    def make_param_names(self):
        """The param_names file lists every parameter's name and Latex tag, and is used for visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are \
        properties of each model class."""
        param_names = open(self.files.param_names, 'w')

        for prior_name, prior_model in self.model_mapper.prior_models:

            param_labels = prior_model.cls.parameter_labels.__get__(prior_model.cls)
            component_number = prior_model.cls().component_number
            subscript = prior_model.cls.subscript.__get__(prior_model.cls) + str(component_number+1)

            param_labels = self.generate_parameter_latex(param_labels, subscript)

            for param_no, param in enumerate(self.model_mapper.class_priors_dict[prior_name]):
                line = prior_name + '_' + param[0]
                line += ' ' * (40 - len(line)) + param_labels[param_no]

                param_names.write(line + '\n')

        param_names.close()

    def output_model_info(self):
        model_info = self.model_mapper.generate_info()
        with open(self.files.model_info, 'w') as file:
            file.write(model_info)
        file.close()

    def check_model_info(self):
        """Check the priors that make up the model_mapper are the same as those which were used to setup the initial \
        MultiNestOptimizer run which has since been terminated."""

        model_info = self.model_mapper.generate_info()

        model_info_check = open(self.files.model_info, 'r')

        if str(model_info_check.read()) != model_info:

            raise MultiNestException(
                'The model_mapper input to MultiNestOptimizer has a different prior for a parameter than the model_mapper existing in '
                'the directory. Parameter = ')

        model_info_check.close()

    def setup_results_intermediate(self):
        return MultiNestResultsIntermediate(self.path, self.obj_name, self.model_mapper)

    def setup_results_final(self):
        return MultiNestResultsFinal(self.path, self.obj_name, self.model_mapper)

class MultiNestResultsIntermediate(NonLinearDirectory):

    def __init__(self, path, obj_name, model_mapper):
        """Class which stores the intermediate results of a MultiNest analysis, e.g. before the non-linear sampling \
        is complete.

        This corresponds to the most likely and most probable models, and allows the model images, residuals, \
        chi-sq image, etc. to be inspected before a MultiNest analysis is complete.

        Parameters
        -----------
        path : str
            The path where the non_linear results are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : CalibrationModel.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        """

        super(MultiNestResultsIntermediate, self).__init__(path, obj_name, model_mapper)

        self.files = MultiNestFiles(self.obj_name, self.results_path)

        self.setup_most_likely_and_probable()

    def setup_most_likely_and_probable(self):
        """Setup the most likely and probable models. This is performed as both 1D vectors of all parameters and a \
        model_mapper instance of each model.

        The most probable model is defined as the model where each parameter is the mean value of all posterior \
        samples weighted by their sampling probabilities.

        The most likely model is defined as the model which gives the highest likelihood, regardless of the inferred
        posterior distribution.
        """
        self._most_probable = self.read_most_probable()
        self._most_likely = self.read_most_likely()

        self.most_probable = self.model_mapper.from_physical_vector(self._most_probable)
        self.most_likely = self.model_mapper.from_physical_vector(self._most_likely)

    def read_most_probable(self):
        return self.read_vector_from_summary(self.files.summary, self.total_parameters, 0)

    def read_most_likely(self):
        return self.read_vector_from_summary(self.files.summary, self.total_parameters, 28)

    @staticmethod
    def read_vector_from_summary(filename_summary, total_parameters, offset):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which results from a \
        multinest analysis.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        Parameters
        -----------
        filename_summary : str
            The directory and file name of the summary file.
        total_parameters : int
            The total number of parameters of the model.
        offset : int
            The summary file stores the most likely model in the first half of columns and the most probable model in
            the second half. The offset is used to start the parsing at the appropriate column.
        """
        summary = open(filename_summary)

        skip = summary.read(2 + offset*total_parameters)  # skip the first 3 characters of the file (indentation)

        vector = []

        for param in range(total_parameters):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector


class MultiNestResultsFinal(MultiNestResultsIntermediate):

    def __init__(self, path, obj_name, model_mapper, limit=0.9973):
        """Class which stores the final results of a MultiNest analysis, including the intermediate results \
        produced by the parent class *MultiNestResultsFinal* (these are the most likely / probably models).

        The final results use the library *GetDist* to compute and visualize the probably distribution function \
        (PDF) of parameter space. This uses the weighted-samples output by MultiNest and allows the marginalized PDF's \
        of parameters in 1D and 2D to be plotted.

        Confidence limits on parameters are also calculated, by marginalized parameters over 1 or 2 dimensions.

        Parameters
        -----------
        path : str
            The path where the non_linear results are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : CalibrationModel.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        limit : float
            The fraction of a PDF used to estimate errors.
        """

        super(MultiNestResultsFinal, self).__init__(path, obj_name, model_mapper)

        self.pdf = getdist.mcsamples.loadMCSamples(self.files.weighted_samples)

        self.setup_1d_upper_and_lower_limits(limit)

    def setup_1d_upper_and_lower_limits(self, limit):
        """Setup 1D vectors of the upper and lower limits of the multinest results.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        limit : float
            The fraction of a PDF used to estimate errors.
        """
        self.densities_1d = list(map(lambda p : self.pdf.get1DDensity(p), self.pdf.getParamNames().names))

        limits = list(map(lambda p : p.getLimits(limit), self.densities_1d))

        self._lower_limits_1d = list(map(lambda p : p[0], limits))
        self._upper_limits_1d = list(map(lambda p : p[1], limits))

    def setup_weighted_sample_model(self, index):
        """Setup a model instance of a weighted sample, including its weight and likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        model, weight, likelihood = self.read_weighted_sample_model(index)

        self._weighted_sample_model = model

        self.weighted_sample_model = self.model_mapper.from_physical_vector(model)
        self.weighted_sample_weight = weight
        self.weighted_sample_likelihood = likelihood

    def read_weighted_sample_model(self, index):
        """From a weighted sample return the model, weight and likelihood hood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        return list(self.pdf.samples[index]), self.pdf.weights[index], -0.5*self.pdf.loglikes[index]


class MultiNestException(Exception):
    pass