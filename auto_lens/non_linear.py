import getdist
import sys
import os

class NonLinearDirectory(object):

    def __init__(self, path, model_mapper):
        """Abstract base class for non-linear optimizers.

        Sets up the directory structure for their results.

        Parameters
        ------------
        path : str
            The path where the non-linear analysis results are stored.
        model_mapper : auto_lens.model_mapper.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        """
        self.path = path
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

        self.results_path = self.path
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

    def __init__(self, results_path):
        """Setup the path and file names of the MultiNestOptimizer files.

        Parameters
        ------------
        results_path : str
            The results path of the MultiNestOptimizer analysis.
        """
        self.param_names = results_path + 'model.paramnames'
        self.model_info = results_path + 'model.info'
        self.weighted_samples = results_path + 'weighted_samples.txt'
        self.summary = results_path + 'summary.txt'


class MultiNestOptimizer(NonLinearDirectory):

    def __init__(self, path, model_mapper):
        """Class to setup and run a MultiNestOptimizer analysis and output the results file.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances passed \
        to each iteration of MultiNestOptimizer.

        Parameters
        ------------
        path : str
            The path where the non_linear results are stored.
        model_mapper : auto_lens.model_mapper.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        """

        super(MultiNestOptimizer, self).__init__(path, model_mapper)

        self.files = MultiNestFiles(self.results_path)

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

    def get_model_info(self):
        """Use the priors that make up the model_mapper to information on each parameter of the overall model.

        This information is extracted from each priors *model_info* property.
        '"""

        model_info = ''

        for prior_name, prior_model in self.model_mapper.prior_models:

            model_info += prior_model.cls.__name__ + '\n' + '\n'

            for i, prior in enumerate(prior_model.priors):
                param_name = str(self.model_mapper.class_priors_dict[prior_name][i][0])
                model_info += param_name + ': ' + (prior[1].model_info + '\n')

            model_info +='\n'

        return model_info

    def output_model_info(self):
        model_info = self.get_model_info()
        file = open(self.files.model_info, 'w')
        file.write(model_info)
        file.close()

    def check_model_info(self):
        """Check the priors that make up the model_mapper are the same as those which were used to setup the initial \
        MultiNestOptimizer run which has since been terminated."""

        model_info = self.get_model_info()

        model_info_check = open(self.files.model_info, 'r')

        if str(model_info_check.read()) != model_info:

            raise MultiNestException(
                'The model_mapper input to MultiNestOptimizer has a different prior for a parameter than the model_mapper existing in '
                'the directory. Parameter = ')

        model_info_check.close()

    def setup_results(self):
        return MultiNestResults(self.path, self.model_mapper)


class MultiNestResults(NonLinearDirectory):

    def __init__(self, path, model_mapper):

        super(MultiNestResults, self).__init__(path, model_mapper)

        self.files = MultiNestFiles(self.results_path)

        # self.pdf = getdist.mcsamples.loadMCSamples(self.files.weighted_samples)

        self._most_probable = self.read_most_probable()
        self._most_likely = self.read_most_likely()

        self.most_probable = self.model_mapper.from_physical_vector(self._most_probable)
        self.most_likely = self.model_mapper.from_physical_vector(self._most_likely)

        # limits = list(map(lambda p : pdf.confidence(paramVec=p, limfrac=limfrac, upper=True),
        #                         range(0, self.total_parameters)))

    def read_most_probable(self):
        """
        Read the most probable model_mapper from the 'summary.txt' file resulting from a non-linear non_linear analysis.

        This file stores the parameters of the most probable model_mapper in its first half of entries.

        The most probable model_mapper is defined as the model_mapper which is the mean value of all samplings of a parameter \
        weighted by their sampling probabilities.

        """

        summary = open(self.files.summary)

        skip = summary.read(2)  # skip the first 3 characters of the file, which are an indentation

        most_probable_vector = []

        for param in range(self.total_parameters):
            most_probable_vector.append(float(summary.read(28)))

        summary.close()

        return most_probable_vector

    def read_most_likely(self):
        """
        Read the most likely model_mapper from the 'summary.txt' file resulting from a non-linear non_linear analysis.

        This file stores the parameters of the most likely model_mapper in its second half of entries.

        The most likely model_mapper is defined as the model_mapper which gives the highest likelihood, regardless of the inferred \
        posterior distributions.

        """

        summary = open(self.files.summary)

        skip = summary.read(2 + 28*self.total_parameters)  # skip the first 3 characters of the file, which are an indentation

        most_likely_vector = []

        for param in range(self.total_parameters):
            most_likely_vector.append(float(summary.read(28)))

        summary.close()

        return most_likely_vector


class MultiNestException(Exception):
    pass