import inspect

from autolens.autofit import model_mapper as mm


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

    def __init__(self, variable_items, constant_items):
        self.variable_items = variable_items
        self.constant_items = constant_items

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

    def __setattr__(self, key, value):
        if key not in ("variable_items", "constant_items"):
            print(key)
            value.mapping_name = key
            for index, obj in enumerate(self):
                if obj.mapping_name == key:
                    self[index] = value
                    return
        super().__setattr__(key, value)


class PhasePropertyList(PhaseProperty):
    def fget(self, obj):
        return ListWrapper(getattr(obj.optimizer.variable, self.name),
                           getattr(obj.optimizer.constant, self.name))

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
        else:
            for n in range(len(value)):
                if inspect.isclass(value[n]):
                    raise AssertionError(
                        "Classes must be wrapped in PriorModel instances to be used in PhasePropertyLists")
                value[n].position = n
        setattr(obj.optimizer.variable, self.name, [item for item in value if is_prior(item)])
        setattr(obj.optimizer.constant, self.name, [item for item in value if not is_prior(item)])
