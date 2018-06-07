import getdist
from auto_lens import exc
import os


def generate_parameter_latex(parameters, subscript=''):
    """Generate a latex label for a non-linear search parameter.

    This is used for the paramnames file and outputting the files of a run to a latex table.

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


class NonLinearFiles(object):

    def __init__(self, path, obj_name, model_mapper, check_model=True):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer files, which are standardized across all \
        non-linear optimizers.

        Parameters
        ------------
        path : str
            The path where the non-linear analysis files are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : model_mapper.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        check_model : bool
            Check whether the model.info file corresponds to the model_mapper passed in.
        """

        self.path = path
        self.obj_name = obj_name
        self.model_mapper = model_mapper
        self.total_parameters = len(self.model_mapper.priors_ordered_by_id)

        self.results_path = self.path + self.obj_name + '/'
        for prior_name, prior_model in self.model_mapper.prior_models:
            self.results_path += prior_model.cls.__name__ + '+'
        self.results_path = self.results_path[:-1] + '/'  # remove last + symbol from path name

        self.file_param_names = self.results_path + self.obj_name + '.paramnames'
        self.file_model_info = self.results_path + 'model.info'

        self.resume = os.path.exists(self.results_path)  # resume True if results path already exists

        if self.resume == False:

            os.makedirs(self.results_path)  # Create results folder if doesnt exist
            self.create_param_names()
            self.model_mapper.output_model_info(self.file_model_info)

        elif self.resume == True:
            if check_model == True:
                self.model_mapper.check_model_info(self.file_model_info)

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


class MultiNest(NonLinearFiles):

    def __init__(self, path, obj_name, model_mapper, check_model=True):
        """Class to setup and run a MultiNest analysis and output the MultInest files.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances that \
        are passed to each iteration of MultiNest.

        Parameters
        ------------
        path : str
            The path where the non_linear files are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : model_mapper.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        """

        super(MultiNest, self).__init__(path, obj_name, model_mapper, check_model)

        self.file_summary = self.results_path + 'summary.txt'

    def open_summary_file(self):

        summary = open(self.file_summary)

        expected_parameters = (len(summary.readline()) - 57) / 56
        if expected_parameters != self.total_parameters:
            raise exc.MultiNestException(
                'The file_summary file has a different number of parameters than the input model')

        return summary

    def read_vector_from_summary(self, number_entries, offset):

        summary = self.open_summary_file()

        summary.seek(0)
        summary.read(2 + offset * self.total_parameters)
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
        return self.read_vector_from_summary(number_entries=self.total_parameters, offset=0)

    def compute_most_likely(self):
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
        return self.read_vector_from_summary(number_entries=self.total_parameters, offset=28)

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

    def create_multinest_finished(self, check_model=True):
        return MultiNestFinished(self.path, self.obj_name, self.model_mapper, check_model)


class MultiNestFinished(MultiNest):

    def __init__(self, path, obj_name, model_mapper, check_model=True):
        """Class which stores the final files of a MultiNest analysis, including the intermediate files \
        produced by the parent class *MultiNestFinished* (these are the most likely / probably models).

        The final files use the library *GetDist* to compute and visualize the probably distribution function \
        (PDF) of parameter space. This uses the weighted-samples output by MultiNest and allows the marginalized PDF's \
        of parameters in 1D and 2D to be plotted.

        Confidence limits on parameters are also calculated, by marginalized parameters over 1 or 2 dimensions.

        Parameters
        -----------
        path : str
            The path where the non_linear files are stored.
        obj_name : str
            Unique identifier of the data being analysed (e.g. the name of the data set)
        model_mapper : CalibrationModel.ModelMapper
            Maps the model priors to a set of parameters (a model instance)
        limit : float
            The fraction of a PDF used to estimate errors.
        """

        super(MultiNestFinished, self).__init__(path, obj_name, model_mapper, check_model)

        self.file_weighted_samples = self.results_path + self.obj_name + '.txt'
        self.pdf = getdist.mcsamples.loadMCSamples(self.file_weighted_samples)

    def compute_model_at_limit(self, limit):
        densities_1d = list(map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names))
        return list(map(lambda p: p.getLimits(limit), densities_1d))

    def compute_model_at_upper_limit(self, limit):
        """Setup 1D vectors of the upper and lower limits of the multinest files.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        limit : float
            The fraction of a PDF used to estimate errors.
        """
        return list(map(lambda param: param[1], self.compute_model_at_limit(limit)))

    def compute_model_at_lower_limit(self, limit):
        """Setup 1D vectors of the upper and lower limits of the multinest files.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        limit : float
            The fraction of a PDF used to estimate errors.
        """
        self.compute_model_at_limit(limit)
        return list(map(lambda param: param[0], self.compute_model_at_limit(limit)))

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

    # TODO : untested and unfinished, remiains to be seen if we'll need this code.

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

        new_summary_file = open(self.results_path + 'summary_new.txt', 'w')
        new_summary_file.write(most_probable + most_likely + likelihood + log_likelihood)
        new_summary_file.close()
