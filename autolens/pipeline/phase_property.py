import inspect

from autolens.autopipe import model_mapper as mm


def is_prior(value):
    return inspect.isclass(value) or isinstance(value, mm.AbstractPriorModel)


class PhaseProperty(object):
    def __init__(self, name):
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

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        raise AttributeError("can't delete attribute")

    def getter(self, fget):
        return type(self)(fget, self.fset, None, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, None, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


class ListWrapper(object):
    def __init__(self, variable_items, constant_items):
        self.variable_items = variable_items
        self.constant_items = constant_items

    def __setitem__(self, i, value):
        pass

    def __getitem__(self, i):
        return sorted(self.variable_items + self.constant_items, key=lambda item: item.position)[i]

    def __eq__(self, other):
        return [item for item in self] == other


class PhasePropertyList(PhaseProperty):
    def fget(self, obj):
        return ListWrapper(getattr(obj.optimizer.variable, self.name),
                           getattr(obj.optimizer.constant, self.name))

    def fset(self, obj, value):
        for n in range(len(value)):
            value[n].position = n
        setattr(obj.optimizer.variable, self.name, [item for item in value if is_prior(item)])
        setattr(obj.optimizer.constant, self.name, [item for item in value if not is_prior(item)])
