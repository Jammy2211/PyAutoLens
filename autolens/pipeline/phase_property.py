import inspect

from autofit.core import model_mapper as mm


def is_prior(value):
    return inspect.isclass(value) or isinstance(value, mm.AbstractPriorModel)


class PhaseProperty(object):
    def __init__(self, name):
        """
        A phase property is a named property of a phase in a pipeline. It implemented setters and getters that allow
        it to associated values with the constant or variable object depending on the type of those values. Note that
        this functionality may be better handled by the model mapper.

        Parameters
        ----------
        name: str
            The name of this property

        Examples
        --------
        >>> class Phase:
        >>>     my_property = PhaseProperty("my_property")
        >>>     def __init__(self, my_property):
        >>>         self.my_property = my_property
        """
        self.name = name

    def fget(self, obj):
        if hasattr(obj.optimizer.constant, self.name):
            return getattr(obj.optimizer.constant, self.name)
        elif hasattr(obj.optimizer.variable, self.name):
            return getattr(obj.optimizer.variable, self.name)

    def fset(self, obj, value):
        if is_prior(value):
            setattr(obj.optimizer.variable, self.name, value)
            try:
                delattr(obj.optimizer.constant, self.name)
            except AttributeError:
                pass
        else:
            setattr(obj.optimizer.constant, self.name, value)
            try:
                delattr(obj.optimizer.variable, self.name)
            except AttributeError:
                pass

    def fdel(self, obj):
        try:
            delattr(obj.optimizer.constant, self.name)
        except AttributeError:
            pass

        try:
            delattr(obj.optimizer.variable, self.name)
        except AttributeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)

    def __set__(self, obj, value):
        self.fset(obj, value)

    def __delete__(self, obj):
        return self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


class ListWrapper(object):
    """
    @DynamicAttrs
    """

    def __init__(self, optimizer, name):
        """
        A ListWrapper takes lists of variable and constants and behaves like a list. Items can be addressed by their
        index in the list or by their name.

        Parameters
        ----------

        """
        self.optimizer = optimizer
        self.variable_items = getattr(optimizer.variable, name)
        self.constant_items = getattr(optimizer.constant, name)

    def __setitem__(self, i, value):
        original = self[i]
        value.position = original.position
        if original in self.variable_items:
            self.variable_items.remove(original)
        if original in self.constant_items:
            self.constant_items.remove(original)
        if is_prior(value):
            self.variable_items.append(value)
        else:
            self.constant_items.append(value)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def items(self):
        return sorted(self.variable_items + self.constant_items, key=lambda item: item.position)

    def __eq__(self, other):
        return list(self.items) == other

    def __len__(self):
        return len(self.items)

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            for obj in self:
                if obj.mapping_name == item:
                    return obj
        raise AttributeError()

    def __setattr__(self, key, value):
        if key not in ("variable_items", "constant_items", "optimizer"):
            setattr(self.optimizer.variable, key, value)
            value.mapping_name = key
            for index, obj in enumerate(self):
                if obj.mapping_name == key:
                    self[index] = value
                    return
        super().__setattr__(key, value)


class PhasePropertyCollection(PhaseProperty):
    """
    A phase property that wraps a list or dictionary. If wrapping a dictionary then named items can still be addressed
    using indexes; if wrapping a list then items can still be addressed by names of the format name_index where name is
    the name of this phase property.
    """

    def fget(self, obj):
        return ListWrapper(obj.optimizer, self.name)

    def fset(self, obj, value):
        if isinstance(value, dict):
            dictionary = value
            value = []
            for n, tup in enumerate(dictionary.items()):
                if inspect.isclass(tup[1]):
                    raise AssertionError(
                        "Classes must be wrapped in PriorModel instances to be used in PhasePropertyLists")
                value.append(tup[1])
                value[n].mapping_name = tup[0]
                value[n].position = n
                if is_prior(tup[1]):
                    setattr(obj.optimizer.variable, tup[0], tup[1])
                else:
                    setattr(obj.optimizer.constant, tup[0], tup[1])
        else:
            for n in range(len(value)):
                if inspect.isclass(value[n]):
                    raise AssertionError(
                        "Classes must be wrapped in PriorModel instances to be used in PhasePropertyLists")
                value[n].mapping_name = "{}_{}".format(self.name, n)
                value[n].position = n
        setattr(obj.optimizer.variable, self.name, [item for item in value if is_prior(item)])
        setattr(obj.optimizer.constant, self.name, [item for item in value if not is_prior(item)])
