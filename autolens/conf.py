import configparser
import os

from autolens import exc


def family(current_class):
    yield current_class
    for next_class in current_class.__bases__:
        for val in family(next_class):
            yield val


class NamedConfig(object):
    """Parses generic config"""

    def __init__(self, config_path):
        """
        Parameters
        ----------
        config_path: String
            The path to the config file
        """
        self.path = config_path
        self.parser = configparser.ConfigParser()
        self.parser.read(self.path)

    def get(self, section_name, attribute_name, attribute_type=str):
        """

        Parameters
        ----------
        section_name
        attribute_type: type
            The type to which the value should be cast
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        prior_array: []
            An array describing a prior
        """
        string_value = self.parser.get(section_name, attribute_name)
        if string_value == "None":
            return None
        if attribute_type is bool:
            return string_value == "True"
        return attribute_type(string_value)

    def has(self, section_name, attribute_name):
        """
        Parameters
        ----------
        section_name
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        has_prior: bool
            True iff a prior exists for the module, class and attribute
        """
        return self.parser.has_option(section_name, attribute_name)


class LabelConfig(NamedConfig):
    def label(self, name):
        return self.get("label", name)

    def subscript(self, cls):
        for family_cls in family(cls):
            if self.has("subscript", family_cls.__name__):
                return self.get("subscript", family_cls.__name__)

        ini_filename = cls.__module__.split(".")[-1]
        raise exc.PriorException(
            "The prior config at {}/{} does not a subscript for {} or any of its parents".format(self.path,
                                                                                                 ini_filename,
                                                                                                 cls.__name__
                                                                                                 ))


class AncestorConfig(object):
    """Parses prior config"""

    def __init__(self, config_folder_path):
        """
        Parameters
        ----------
        config_folder_path: String
            The path to the prior config folder
        """
        self.path = config_folder_path
        self.parser = configparser.ConfigParser()

    def read(self, module_name):
        """
        Read a particular config file

        Parameters
        ----------
        module_name: String
            The analysis_path of the module for which a config is to be read (priors relate one to one with configs).
        """
        self.parser.read("{}/{}.ini".format(self.path, module_name.split(".")[-1]))

    def get_for_nearest_ancestor(self, cls, attribute_name):
        """
        Find a prior with the attribute analysis_path from the config for this class or one of its ancestors

        Parameters
        ----------
        cls: class
            The class of interest
        attribute_name: String
            The analysis_path of the attribute
        Returns
        -------
        prior_array: []
            An array describing this prior
        """
        for family_cls in family(cls):
            if self.has(family_cls.__module__, family_cls.__name__, attribute_name):
                return self.get(family_cls.__module__, family_cls.__name__, attribute_name)

        ini_filename = cls.__module__.split(".")[-1]
        raise exc.PriorException(
            "The prior config at {}/{} does not contain {} in {} or any of its parents".format(self.path,
                                                                                               ini_filename,
                                                                                               attribute_name,
                                                                                               cls.__name__
                                                                                               ))

    def get(self, module_name, class_name, attribute_name):
        """

        Parameters
        ----------
        module_name: String
            The analysis_path of the module
        class_name: String
            The analysis_path of the class
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        prior_array: []
            An array describing a prior
        """
        self.read(module_name)
        return self.parser.get(class_name, attribute_name)

    def has(self, module_name, class_name, attribute_name):
        """
        Parameters
        ----------
        module_name: String
            The analysis_path of the module
        class_name: String
            The analysis_path of the class
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        has_prior: bool
            True iff a prior exists for the module, class and attribute
        """
        self.read(module_name)
        return self.parser.has_option(class_name, attribute_name)


class DefaultPriorConfig(AncestorConfig):
    """Parses prior config"""

    def get(self, module_name, class_name, attribute_name):
        """

        Parameters
        ----------
        module_name: String
            The analysis_path of the module
        class_name: String
            The analysis_path of the class
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        prior_array: []
            An array describing a prior
        """
        arr = super(DefaultPriorConfig, self).get(module_name, class_name, attribute_name).replace(" ", "").split(",")
        return [arr[0]] + list(map(float, arr[1:]))


class LimitConfig(AncestorConfig):
    """Parses prior config"""

    def get(self, module_name, class_name, attribute_name):
        """

        Parameters
        ----------
        module_name: String
            The analysis_path of the module
        class_name: String
            The analysis_path of the class
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        prior_limits: ()
            A tuple containing the limits of the range of values an attribute can take with an exception being thrown
            if a nonlinear search produces a value outside of that range.
        """
        arr = super(LimitConfig, self).get(module_name, class_name, attribute_name).replace(" ", "").split(",")
        return tuple(map(float, arr[:2]))


class WidthConfig(AncestorConfig):
    def get(self, module_name, class_name, attribute_name):
        """

        Parameters
        ----------
        module_name: String
            The analysis_path of the module
        class_name: String
            The analysis_path of the class
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        width: float
            The default width of a gaussian prior for this attribute
        """
        return float(super(WidthConfig, self).get(module_name, class_name, attribute_name))


class Config(object):
    def __init__(self, config_path, output_path):
        self.config_path = config_path
        self.prior_default = DefaultPriorConfig("{}/priors/default".format(config_path))
        self.prior_width = WidthConfig("{}/priors/width".format(config_path))
        self.limit_config = LimitConfig("{}/priors/limit".format(config_path))
        self.non_linear = NamedConfig("{}/non_linear.ini".format(config_path))
        self.label = LabelConfig("{}/label.ini".format(config_path))
        self.general = NamedConfig("{}/general.ini".format(config_path))
        self.output_path = output_path


def is_config_in(folder):
    return os.path.isdir("{}/config".format(folder))


"""
Search for default configuration and put output in the same folder as config.

The search is performed in this order:
1) workspace. This is assumed to be in the same directory as autolens in the Docker container
2) current working directory. This is to allow for installation and use with pip where users would expect the
   configuration in their current directory to be used.
3) autolens. This is a backup for when no configuration is found. In this case it is still assumed a workspace directory
   exists in the same directory as autolens.
"""

autolens_directory = os.path.dirname(os.path.realpath(__file__))
workspace_directory = "{}/../workspace".format(autolens_directory)
current_directory = os.getcwd()

if is_config_in(workspace_directory):
    CONFIG_PATH = "{}/config".format(workspace_directory)
    instance = Config(CONFIG_PATH, "{}/output/".format(workspace_directory))
elif is_config_in(current_directory):
    CONFIG_PATH = "{}/config".format(current_directory)
    instance = Config(CONFIG_PATH, "{}/output/".format(current_directory))
else:
    CONFIG_PATH = "{}/config".format(autolens_directory)
    instance = Config(CONFIG_PATH, "{}/../workspace/output/".format(autolens_directory))
