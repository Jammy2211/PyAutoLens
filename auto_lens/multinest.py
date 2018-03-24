import getdist
import sys
import os

class MultiNest(object):

    def __init__(self, path, model):
        """Class to store the MultiNest analysis results and directory structure.

        This interfaces with an input model, setting up the directory structure based on input profiles and using it \
        for settings up model instances of the MultiNest results.

        Parameters
        ------------
        path : str
            The path where the multinest results are stored.
        model : auto_lens.model.ModelMapper
            The cti model being analysed, including the model setup and priors.
        """

        self.path = path
        self.model = model

        self.setup_results_path()
        self.setup_filenames()

        if self.results_folder_exists() == False:
            self.make_results_folder()
            self.make_param_names()

    def setup_results_path(self):
        """Use the MultiNest model to set up the results path of the MultiNest run."""

        self.results_path = self.path
        for prior_name, prior_model in self.model.prior_models:
            self.results_path += prior_model.cls.__name__
            self.results_path += '+'

        self.results_path = self.results_path[:-1] # remove last + symbol from path name
        self.results_path +='/'

    def setup_filenames(self):
        """Setup the path and file names of the MultiNest files"""
        self.file_param_names = self.results_path + 'model.paramnames'
        self.file_model_info = self.results_path + 'model.info'
        self.file_summary = self.results_path + 'summary.txt'

    def results_folder_exists(self):
        return os.path.exists(self.results_path)

    def make_results_folder(self):
        os.makedirs(self.results_path)

    def make_param_names(self):
        """Make this model's param_names file, which lists every parameter's name and LaTex tag for visualization."""
        param_names = open(self.file_param_names, 'w')

        for prior_name, prior_model in self.model.prior_models:

            param_labels = prior_model.cls.parameter_labels.__get__(prior_model.cls)
            component_number = prior_model.cls().component_number
            subscript = prior_model.cls.subscript.__get__(prior_model.cls) + str(component_number+1)

            param_labels = generate_parameter_latex(param_labels, subscript)

            for param_no, param in enumerate(self.model.class_priors_dict[prior_name]):
                line = prior_name + '_' + param[0]
                line += ' ' * (40 - len(line)) + param_labels[param_no]

                param_names.write(line + '\n')

        param_names.close()

    def setup_most_probable_model_instance(self):
        """Setup a model instance of the most probable model"""
        return self.model.model_instance_from_physical_vector(self.read_most_probable())

    def setup_most_likely_model_instance(self):
        """Setup a model instance of the most likely model"""
        return self.model.model_instance_from_physical_vector(self.read_most_likely())

    def read_most_probable(self):
        """
        Read the most probable model from the 'summary.txt' file resulting from a non-linear multinest analysis.

        This file stores the parameters of the most probable model in its first half of entries.

        The most probable model is defined as the model which is the mean value of all samplings of a parameter \
        weighted by their sampling probabilities.

        """

        summary = open(self.file_summary)

        total_parameters = compute_total_parameters(summary)

        skip = summary.read(2)  # skip the first 3 characters of the file, which are an indentation

        most_probable_vector = []

        for param in range(total_parameters):
            most_probable_vector.append(float(summary.read(28)))

        summary.close()

        return most_probable_vector

    def read_most_likely(self):
        """
        Read the most likely model from the 'summary.txt' file resulting from a non-linear multinest analysis.

        This file stores the parameters of the most likely model in its second half of entries.

        The most likely model is defined as the model which gives the highest likelihood, regardless of the inferred \
        posterior distributions.

        """

        summary = open(self.file_summary)

        total_parameters = compute_total_parameters(summary)

        skip = summary.read(2 + 28*total_parameters)  # skip the first 3 characters of the file, which are an indentation

        most_likely_vector = []

        for param in range(total_parameters):
            most_likely_vector.append(float(summary.read(28)))

        summary.close()

        return most_likely_vector

def generate_parameter_latex(parameters, subscript=''):
    """Generate a latex label for a parameter, typically used for the MultiNest / getdist paramnames file and \
    outputting the results of a MultiNest run to a latex table.

    Parameters
    ----------
    parameters : [str]
        The parameter names to be converted to latex.
    subscript : str
        The subscript of the latex entry, often giving the parameter type (e.g. light or dark matter) or numerical \
        number of the component of the model.

    """

    latex = []

    if subscript == '':
        for param in parameters:
            latex.append('$' + param + '$')
    else:
        for param in parameters:
            latex.append('$' + param + r'_{\mathrm{' + subscript + '}}$')

    return latex


def compute_total_parameters(summary, reset_position=True):
    """ Each parameter in the summary file is 28 characters long (including its 4 spaces). Parameters are listed twice \
    (most probable and most likely models) and there are two extra 28 character slots for the most likley model's \
    likelihood and loglikelihood.

    Therefore, the total parameters be computed from the summary file by diving the length of its top line by 28, \
    halving this value and subtracting 2."""

    total_parameters = ((len(summary.readline()) // 28) // 2) - 1
    if reset_position == True:
        summary.seek(0)

    return total_parameters
