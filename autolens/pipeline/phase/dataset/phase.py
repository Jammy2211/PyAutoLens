from autoconf import conf
from os import path
import autoarray as aa
import autogalaxy as ag
from autolens.fit import fit_positions
from autogalaxy.pipeline.phase import dataset
from autolens.pipeline.phase.extensions.stochastic_phase import StochasticPhase
from autolens import exc
import pickle
import json
import numpy as np

import copy


class PhaseDataset(dataset.PhaseDataset):
    def run(
        self,
        dataset,
        mask,
        results=None,
        info=None,
        pickle_files=None,
        log_likelihood_cap=None,
    ):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask2D
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        dataset: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """

        result = super().run(
            dataset=dataset,
            mask=mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
            log_likelihood_cap=log_likelihood_cap,
        )

        if self.should_output_stochastic_log_evidence_file:

            stochastic_log_evidences = result.stochastic_log_evidences

            self.save_stochastic_log_evidences_to_json(
                stochastic_log_evidences=stochastic_log_evidences
            )

            self.save_stochastic_log_evidences_to_pickle(
                stochastic_log_evidences=stochastic_log_evidences
            )

        try:

            result.analysis.visualizer.visualize_stochastic_histogram(
                paths=self.paths,
                log_evidences=self.load_stochastic_log_evidences_from_json(),
                max_log_evidence=result.log_likelihood,
                histogram_bins=self.settings.settings_lens.stochastic_histogram_bins,
            )

        except FileNotFoundError:

            pass

        return result

    def modify_dataset(self, dataset, results):

        # TODO : There is a very weird error no cosma for this line we don't yet undersatand. This try / except fixes it.

        try:

            positions_copy = copy.copy(dataset.positions)

            positions = self.updated_positions_from_positions_and_results(
                positions=positions_copy, results=results
            )

            dataset.positions = positions

        except OSError:
            pass

    def modify_settings(self, dataset, results):

        positions_threshold = self.updated_positions_threshold_from_positions(
            positions=dataset.positions, results=results
        )
        self.settings.settings_lens = self.settings.settings_lens.modify_positions_threshold(
            positions_threshold=positions_threshold
        )
        self.check_positions(positions=dataset.positions)

        preload_sparse_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )
        self.settings.settings_pixelization = self.settings.settings_pixelization.modify_preload(
            preload_sparse_grids_of_planes=preload_sparse_grids_of_planes
        )

    def updated_positions_from_positions_and_results(self, positions, results):
        """If automatic position updating is on, update the phase's positions using the results of the previous phase's
        lens model, by ray-tracing backwards the best-fit source centre(s) to the image-plane.

        The outcome of this function are as follows:

        1) If auto positioning is off (self.auto_positions_factor is None), use the previous phase's positions.
        2) If auto positioning is on (self.auto_positions_factor not None) use positions based on the previous phase's
           best-fit tracer. However, if this tracer gives 1 or less positions, use the previous positions.
        3) If this previous tracer is composed of only 1 plane (e.g. you are light profile fitting the image-plane
           only), use the previous positions.
        4) If auto positioning is on or off and there is no previous phase, use the input positions.
        """

        if results.last is not None:

            if not hasattr(results.last, "positions"):
                return positions
            try:
                if len(results.last.max_log_likelihood_tracer.planes) <= 1:
                    return positions
            except AttributeError:
                pass

        if (
            self.settings.settings_lens.auto_positions_factor is not None
            and results.last is not None
        ):

            updated_positions = (
                results.last.image_plane_multiple_image_positions_of_source_plane_centres
            )

            # TODO : Coorrdinates refascotr will sort out index call here

            if isinstance(updated_positions, aa.GridIrregularGrouped):
                if updated_positions.in_grouped_list:
                    if len(updated_positions.in_grouped_list[0]) > 1:
                        return updated_positions

        if results.last is not None:
            if results.last.positions is not None:
                if results.last.positions.in_grouped_list:
                    return results.last.positions

        return positions

    def updated_positions_threshold_from_positions(self, positions, results) -> [float]:
        """
        If automatic position updating is on, update the phase's threshold using this phase's updated positions.

        First, we ray-trace forward the positions of the source-plane centres (see above) via the mass model to
        determine how far apart they are separated. This gives us their source-plane sepration, which is multiplied by
        self.auto_positions_factor to set the threshold.

        The threshold is rounded up to the auto positions minimum threshold if that setting is included."""

        if results.last is not None:

            try:
                if len(results.last.max_log_likelihood_tracer.planes) <= 1:
                    if self.settings.settings_lens.positions_threshold is not None:
                        return self.settings.settings_lens.positions_threshold
                    if (
                        self.settings.settings_lens.auto_positions_minimum_threshold
                        is not None
                    ):
                        return (
                            self.settings.settings_lens.auto_positions_minimum_threshold
                        )
                    return None
            except AttributeError:
                pass

        if (
            self.settings.settings_lens.auto_positions_factor
            and results.last is not None
        ):

            if positions is None:
                return None

            positions_fits = fit_positions.FitPositionsSourcePlaneMaxSeparation(
                positions=aa.GridIrregularGrouped(grid=positions),
                tracer=results.last.max_log_likelihood_tracer,
                noise_value=1.0,
            )

            positions_threshold = (
                self.settings.settings_lens.auto_positions_factor
                * np.max(positions_fits.maximum_separations)
            )

        else:

            positions_threshold = self.settings.settings_lens.positions_threshold

        if (
            self.settings.settings_lens.auto_positions_minimum_threshold is not None
            and positions_threshold is not None
        ):
            if (
                positions_threshold
                < self.settings.settings_lens.auto_positions_minimum_threshold
            ) or (positions_threshold is None):
                positions_threshold = (
                    self.settings.settings_lens.auto_positions_minimum_threshold
                )

        return positions_threshold

    def preload_pixelization_grids_of_planes_from_results(self, results):

        if self.is_hyper_phase:
            return None

        if (
            results.last is not None
            and self.pixelization is not None
            and not self.pixelization_is_model
        ):
            if self.pixelization.__class__ is results.last.pixelization.__class__:
                if hasattr(results.last, "hyper_combined"):
                    return (
                        results.last.hyper_combined.max_log_likelihood_pixelization_grids_of_planes
                    )
                else:
                    return results.last.max_log_likelihood_pixelization_grids_of_planes
        return None

    def check_positions(self, positions):

        if (
            self.settings.settings_lens.positions_threshold is not None
            and positions is None
        ):
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

    @property
    def stochastic_log_evidences_json_file(self):
        return path.join(self.paths.output_path, "stochastic_log_evidences.json")

    @property
    def should_output_stochastic_log_evidence_file(self):
        if self.uses_cluster_inversion:
            if not path.exists(self.stochastic_log_evidences_json_file):
                if conf.instance["general"]["hyper"]["stochastic_outputs"]:
                    return True
        return False

    def load_stochastic_log_evidences_from_json(self):
        with open(self.stochastic_log_evidences_json_file, "r") as f:
            return np.asarray(json.load(f))

    def save_stochastic_log_evidences_to_json(self, stochastic_log_evidences):
        """
        Save the dataset associated with the phase
        """
        with open(self.stochastic_log_evidences_json_file, "w") as outfile:
            json.dump(
                [float(evidence) for evidence in stochastic_log_evidences], outfile
            )

    def save_stochastic_log_evidences_to_pickle(self, stochastic_log_evidences):
        """
        Save the dataset associated with the phase
        """
        with open(
            path.join(self.paths.pickle_path, "stochastic_log_evidences.pickle"), "wb"
        ) as f:
            pickle.dump(stochastic_log_evidences, f)

    def extend_with_stochastic_phase(
        self,
        stochastic_search=None,
        include_lens_light=False,
        include_pixelization=False,
        include_regularization=False,
        stochastic_method="gaussian",
        stochastic_sigma=0.0,
    ):

        if stochastic_search is None:
            stochastic_search = self.search.copy_with_name_extension(extension="")

        model_classes = [ag.mp.MassProfile]

        if include_lens_light:
            model_classes.append(ag.lp.LightProfile)

        if include_pixelization:
            model_classes.append(ag.pix.Pixelization)

        if include_regularization:
            model_classes.append(ag.reg.Regularization)

        return StochasticPhase(
            phase=self,
            hyper_search=stochastic_search,
            model_classes=tuple(model_classes),
            stochastic_method=stochastic_method,
            stochastic_sigma=stochastic_sigma,
        )
