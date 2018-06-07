class MockImage:
    pass


class MockPixelization:
    pass


class MockNLO:
    def __init__(self):
        pass

    def run(self, fitness_function, priors):
        pass


class MockPriorModel:
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.centre = "centre for {}".format(name)
        self.phi = "phi for {}".format(name)


class MockModelMapper:
    def __init__(self):
        self.classes = {}

    def add_class(self, name, cls):
        self.classes[name] = cls
        return MockPriorModel(name, cls)


class MockModelInstance:
    pass
