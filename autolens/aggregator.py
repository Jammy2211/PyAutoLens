import os


class PhaseOutput(object):
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

    @property
    def header(self):
        return "/".join((self.pipeline, self.phase, self.lens))

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)


class Aggregator(object):
    def __init__(self, directory):
        self.directory = directory
        self.phases = []

        for root, _, filenames in os.walk(directory):
            if ".metadata" in filenames:
                self.phases.append(PhaseOutput(root))

    def phases_with(self, **kwargs):
        return [phase for phase in self.phases if
                all([getattr(phase, key) == value for key, value in kwargs.items()])]

    def model_results(self, **kwargs):
        return "\n\n".join("{}\n\n{}".format(phase.header, phase.model_results) for phase in
                           self.phases_with(**kwargs))


if __name__ == "__main__":
    from sys import argv

    root_directory = None
    try:
        root_directory = argv[1]
    except IndexError:
        print("Usage\n\naggregator.py (root_directory) [pipeline=pipeline phase=phase lens=lens]")
        exit(1)
    filter_dict = {pair[0]: pair[1] for pair in [line.split("=") for line in argv[2:]]}

    with open("model.results", "w+") as out:
        out.write(Aggregator(root_directory).model_results(**filter_dict))
