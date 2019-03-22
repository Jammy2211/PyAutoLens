import os


class Aggregation(object):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path) as f:
            lines = f.readlines()
            pairs = [line.replace("\n", "").split("=") for line in lines]
            value_dict = {pair[0]: pair[1] for pair in pairs}
            self.pipeline = value_dict["pipeline"]
            self.phase = value_dict["phase"]
            self.lens = value_dict["lens"]


class Aggregator(object):
    def __init__(self, directory):
        self.directory = directory
        self.aggregations = []

        for root, _, filenames in os.walk(directory):
            for filename in filter(lambda f: f == ".metadata", filenames):
                self.aggregations.append(Aggregation(os.path.join(root, filename)))
