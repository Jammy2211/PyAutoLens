import os


class Aggregation(object):
    def __init__(self, directory):
        self.directory = directory
        self.file_path = os.path.join(directory, ".metadata")
        with open(self.file_path) as f:
            self.text = f.read()
            pairs = [line.split("=") for line in self.text.split("\n")]
            value_dict = {pair[0]: pair[1] for pair in pairs}
            self.pipeline = value_dict["pipeline"]
            self.phase = value_dict["phase"]
            self.lens = value_dict["lens"]

    @property
    def model_results(self):
        with open(os.path.join(self.directory, "model.results")) as f:
            return f.read()

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)


class Aggregator(object):
    def __init__(self, directory):
        self.directory = directory
        self.aggregations = []

        for root, _, filenames in os.walk(directory):
            if ".metadata" in filenames:
                self.aggregations.append(Aggregation(root))

    def aggregations_with(self, **kwargs):
        return [aggregation for aggregation in self.aggregations if
                all([getattr(aggregation, key) == value for key, value in kwargs.items()])]
