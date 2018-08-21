import inspect

from autolens.autopipe import model_mapper as mm


def is_prior(value):
    return inspect.isclass(value) or isinstance(value, mm.AbstractPriorModel)


def phase_property(name):
    """
    Create a property that is tied to the non_linear instance determines whether to set itself as a constant or
    variable.

    Parameters
    ----------
    name: str
        The phase_name of this variable

    Returns
    -------
    property: property
        A property that appears to be an attribute of the phase but is really an attribute of constant or variable.
    """

    def fget(self):
        def attribute_from(source, other):
            attribute = getattr(source, name)
            if isinstance(attribute, list):
                return list(sorted(attribute + getattr(other, name), key=lambda item: item.position))
            return attribute

        if hasattr(self.optimizer.constant, name):
            return attribute_from(self.optimizer.constant, self.optimizer.variable)
        elif hasattr(self.optimizer.variable, name):
            return attribute_from(self.optimizer.variable, self.optimizer.constant)

    def fset(self, value):
        if isinstance(value, list):
            for n in range(len(value)):
                value[n].position = n
            setattr(self.optimizer.variable, name, [item for item in value if is_prior(item)])
            setattr(self.optimizer.constant, name, [item for item in value if not is_prior(item)])
        elif is_prior(value):
            setattr(self.optimizer.variable, name, value)
            try:
                delattr(self.optimizer.constant, name)
            except AttributeError:
                pass
        else:
            setattr(self.optimizer.constant, name, value)
            try:
                delattr(self.optimizer.variable, name)
            except AttributeError:
                pass

    return property(fget=fget, fset=fset, doc=name)
