


class MockMassProfile(object):

    def __init__(self, value):
        self.value = value


class MockMapping(object):

    def __init__(self):
        pass


class MockPixelization(object):

    def __init__(self, value):
        self.value = value

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_from_grids_and_borders(self, grids, borders):
        return self.value


class MockRegularization(object):

    def __init__(self, value):
        self.value = value


class MockBorders(object):

    def __init__(self, image=None, sub=None):
        self.image = image
        self.sub = sub