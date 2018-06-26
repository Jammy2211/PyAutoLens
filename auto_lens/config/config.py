import configparser

from auto_lens import exc


class Config(object):
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
            The name of the module for which a config is to be read (priors relate one to one with configs).
        """
        self.parser.read("{}/{}.ini".format(self.path, module_name.split(".")[-1]))

    def get_for_nearest_ancestor(self, cls, attribute_name):
        """
        Find a prior with the attribute name from the config for this class or one of its ancestors

        Parameters
        ----------
        cls: class
            The class of interest
        attribute_name: String
            The name of the attribute
        Returns
        -------
        prior_array: []
            An array describing this prior
        """

        def family(current_class):
            yield current_class
            for next_class in current_class.__bases__:
                for val in family(next_class):
                    yield val

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
            The name of the module
        class_name: String
            The name of the class
        attribute_name: String
            The name of the attribute

        Returns
        -------
        prior_array: []
            An array describing a prior
        """
        self.read(module_name)
        arr = self.parser.get(class_name, attribute_name).replace(" ", "").split(",")
        return [arr[0]] + list(map(float, arr[1:]))

    def has(self, module_name, class_name, attribute_name):
        """
        Parameters
        ----------
        module_name: String
            The name of the module
        class_name: String
            The name of the class
        attribute_name: String
            The name of the attribute

        Returns
        -------
        has_prior: bool
            True iff a prior exists for the module, class and attribute
        """
        self.read(module_name)
        return self.parser.has_option(class_name, attribute_name)