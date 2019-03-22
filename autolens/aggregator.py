import os


class Aggregation(object):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path) as f:
            self.text = f.read()
            pairs = [line.split("=") for line in self.text.split("\n")]
            value_dict = {pair[0]: pair[1] for pair in pairs}
            self.pipeline = value_dict["pipeline"]
            self.phase = value_dict["phase"]
            self.lens = value_dict["lens"]

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)


class Aggregator(object):
    def __init__(self, directory):
        self.directory = directory
        self.aggregations = []

        for root, _, filenames in os.walk(directory):
            for filename in filter(lambda f: f == ".metadata", filenames):
                self.aggregations.append(Aggregation(os.path.join(root, filename)))

    def aggregations_with(self, **kwargs):
        return [aggregation for aggregation in self.aggregations if
                all([getattr(aggregation, key) == value for key, value in kwargs.items()])]
