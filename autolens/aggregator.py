#!/usr/bin/env python

"""
Filter and collate phase data in all subdirectories.

Usage:

./aggregator.py (root_directory) [pipeline=pipeline phase=phase lens=lens]

Example:

./aggregator.py ../output pipeline=lens_mass_x1_source_x1_positions
"""

import os


class PhaseOutput(object):
    def __init__(self, directory: str):
        """
        Represents the output of a single phase. Comprises a .metadata file and other data files.

        Parameters
        ----------
        directory
            The directory of the phase
        """
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
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with open(os.path.join(self.directory, "model.results")) as f:
            return f.read()

    @property
    def header(self) -> str:
        """
        A header created by joining the pipeline, phase and lens names
        """
        return "/".join((self.pipeline, self.phase, self.lens))

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<PhaseOutput {}>".format(self)


class Aggregator(object):
    def __init__(self, directory: str):
        """
        Class to aggregate phase results for all subdirectories in a given directory.

        The whole directory structure is traversed and a Phase object created for each directory that contains a
        .metadata file.

        Parameters
        ----------
        directory
        """
        self.directory = directory
        self.phases = []

        for root, _, filenames in os.walk(directory):
            if ".metadata" in filenames:
                self.phases.append(PhaseOutput(root))

    def phases_with(self, **kwargs) -> [PhaseOutput]:
        """
        Filters phases. If no arguments are passed all phases are returned. Arguments must be key value pairs, with
        phase, lens or pipeline as the key.

        Parameters
        ----------
        kwargs
            Filters, e.g. pipeline=pipeline1
        """
        return [phase for phase in self.phases if
                all([getattr(phase, key) == value for key, value in kwargs.items()])]

    def model_results(self, **kwargs) -> str:
        """
        Collates model results from all phases in the directory or some subset if filters are applied.

        Parameters
        ----------
        kwargs
            Filters, e.g. pipeline=pipeline1

        Returns
        -------
        model_results
            A string joining headers and results for all included phases.
        """
        return "\n\n".join("{}\n\n{}".format(phase.header, phase.model_results) for phase in
                           self.phases_with(**kwargs))


if __name__ == "__main__":
    from sys import argv

    root_directory = None
    try:
        root_directory = argv[1]
    except IndexError:
        print("Usage:\n\naggregator.py (root_directory) [pipeline=pipeline phase=phase lens=lens]")
        exit(1)
    filter_dict = {pair[0]: pair[1] for pair in [line.split("=") for line in argv[2:]]}

    with open("model.results", "w+") as out:
        out.write(Aggregator(root_directory).model_results(**filter_dict))
