import os


class Aggregation(object):
    def __init__(self, file_path):
        self.file_path = file_path


class Aggregator(object):
    def __init__(self, directory):
        self.directory = directory
        self.aggregations = []

        for root, _, filenames in os.walk(directory):
            for filename in filter(lambda f: f == ".metadata", filenames):
                self.aggregations.append(Aggregation(os.path.join(root, filename)))
