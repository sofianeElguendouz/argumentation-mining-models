#!/usr/bin/env python
"""
Script to update Artifacts Tracking URI for MLFlow.

This script intended purpose is to take a local run (logged in a directory via
MLFlow) and update the URIs of the Artifact locations. This is useful when, for
example, the training/evaluation scripts were run on a remote server, but the
only way to access the MLFlow UI is moving that results to a local environment.

    Argumentation Mining Transformers Module Training Script
    Copyright (C) 2024 Cristian Cardellino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import logging
import yaml

from mlflow import MlflowClient
from pathlib import Path
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mlflow-uri",
        required=True,
        type=Path,
        help="Path to the directory where the MLFlow experiments are.",
    )

    args = parser.parse_args()
    mlflow_uri = args.mlflow_uri  # type: Path

    logger.info("Loading MLFlow client.")
    client = MlflowClient(mlflow_uri.absolute().as_uri())

    logger.info("Updating experiments artifacts locations and URIs.")
    for experiment in tqdm(client.search_experiments()):
        experiment_meta_file = mlflow_uri / experiment.experiment_id / "meta.yaml"
        with open(experiment_meta_file, "rt") as fh:
            meta = yaml.load(fh, Loader=yaml.SafeLoader)
        new_artifact_location = mlflow_uri / experiment.experiment_id
        meta["artifact_location"] = new_artifact_location.absolute().as_uri()
        with open(experiment_meta_file, "wt") as fh:
            yaml.dump(meta, fh, Dumper=yaml.SafeDumper)

        for run in client.search_runs(experiment.experiment_id):
            run_meta_file = mlflow_uri / experiment.experiment_id / run.info.run_id / "meta.yaml"
            with open(run_meta_file, "rt") as fh:
                meta = yaml.load(fh, Loader=yaml.SafeLoader)
            new_artifact_uri = mlflow_uri / experiment.experiment_id / run.info.run_id / "artifacts"
            meta["artifact_uri"] = new_artifact_uri.absolute().as_uri()
            with open(run_meta_file, "wt") as fh:
                yaml.dump(meta, fh, Dumper=yaml.SafeDumper)
