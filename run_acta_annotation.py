"""
Annotation script for ACTA Module. Given a plain text file with the data to
annotate it will run an annotation pipeline according to configuration
arguments.

   Copyright 2023 The ANTIDOTE Project Contributors <https://univ-cotedazur.eu/antidote>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import argparse
import json
import logging
import sys

from itertools import permutations
from pathlib import Path

from acta.models import RelationClassificationTransformerModule, SequenceTaggingTransformerModule
from acta.pipeline import relation_classification, sequence_tagging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We have different option subparsers based on the
    subparsers = parser.add_subparsers(title="pipelines", dest="pipeline")

    # Base parser with common options
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--input-file",
                             required=True,
                             type=Path,
                             help="Path to the file to annotate. Should be a plain text file. "
                                  "WARNING: The time for annotation will increase exponentially "
                                  "the larger the file is.")
    base_parser.add_argument("--output-file",
                             required=True,
                             type=Path,
                             help="Path to the output file. Must be a JSON file. It if exists the "
                                  "script will load it first and will overwrite only the parts of "
                                  "the pipeline that are run.")
    base_parser.add_argument("--max-seq-length",
                             type=int,
                             help="The maximum total input sequence length after tokenization."
                                  "Sequences longer than this will be truncated, sequences shorter "
                                  "will be padded. If left empty it will truncate to the model's "
                                  "max size and pad to the maximum size of each training step.")
    base_parser.add_argument("--truncation-strategy",
                             choices=['longest_first', 'only_first', 'only_second',
                                      'do_not_truncate'],
                             default='do_not_truncate',
                             help="What truncation strategy to use. Must be one of "
                                  "`longest_first`, `only_second`, `only_first` or "
                                  "`do_not_truncate`. "
                                  "Check https://huggingface.co/docs/transformers/pad_truncation "
                                  "for more information.")

    arg_comp_parser = subparsers.add_parser("arg", parents=[base_parser],
                                            help="Runs the Argumentative Components Detection "
                                                 "pipeline.")
    arg_comp_parser.add_argument("--arg-comp-model", "-a",
                                 metavar="ARGUMENTATIVE_COMPONENTS_MODEL_PATH",
                                 required=True,
                                 type=Path,
                                 help="Path to the `acta.models.SequenceTaggingTransformerModule`, "
                                      "trained for Argumentative Components Detection.")

    rel_class_parser = subparsers.add_parser("rel", parents=[arg_comp_parser],
                                             add_help=False,
                                             help="Runs the Argumentative Relationship "
                                                  "Classification pipeline.")
    rel_class_parser.add_argument("--rel-class-model", "-r",
                                  metavar="RELATION_CLASSIFICATION_MODEL_PATH",
                                  required=True,
                                  type=Path,
                                  help="Path to the `RelationClassificationTransformerModule`, "
                                       "trained for Argumentative Relationship Classification.")
    rel_class_parser.add_argument("--filter-no-rel-class",
                                  action="store_true",
                                  help="If active removes the 'No Relationship' predictions from "
                                       "the final annotations.")
    rel_class_parser.add_argument("--no-rel-class",
                                  default="noRel",
                                  help="The class that defines that there's no relationship.")
    rel_class_parser.add_argument("--confidence",
                                  action="store_true",
                                  help="If active, adds confidence to the argumentative structure "
                                       "predictions.")
    rel_class_parser.add_argument("--confidence-as-probability",
                                  action="store_true",
                                  help="If active, the confidence is the probability.")

    args = parser.parse_args()

    if args.output_file.exists():
        logger.warning(f"The output file {args.output_file} exists. Trying to load it.")
        try:
            with open(args.output_file, "rt") as fh:
                annotations = json.load(fh)
        except BaseException as e:
            logger.error(f"There was a problem loading the output file: {e}. "
                         "Please remove it or choose a different file.")
            sys.exit(1)
    else:
        annotations = {}

    with open(args.input_file, "rt") as fh:
        input_text = ' '.join(fh.read().split())  # Remove double whitespaces and newlines

    if args.pipeline in {'arg', 'rel'}:
        logger.info("Loading argumentative components detection model.")
        # Both for the commands of `arg` and `rel` we need to identify the argumentative components
        arg_comp_model = SequenceTaggingTransformerModule.load_from_checkpoint(
            args.arg_comp_model
        )

        # FIXME: We use the default tokenizer and id2label present in the config
        logger.info("Tagging argumentative components.")
        arg_comp_annotations, arg_comp_relevant = sequence_tagging(
            text=input_text,
            model=arg_comp_model,
            tokenizer=arg_comp_model.config.name_or_path,
            id2label=arg_comp_model.config.id2label,
            max_seq_length=args.max_seq_length,
            truncation_strategy=args.truncation_strategy
        )

        # From the list of relevant argumentative components we build a list of
        # nodes with ids that start from 0 and the reference being the index in
        # the full list of annotations, in this way we avoid redundancy of data
        arg_comp = [{'id': idx, 'ref': ref} for idx, ref in enumerate(arg_comp_relevant)]

        annotations['argumentative_components'] = {
            'full': arg_comp_annotations,
            'relevant': arg_comp
        }

    if args.pipeline == 'rel':
        logger.info("Loading relation classification model.")
        rel_class_model = RelationClassificationTransformerModule.load_from_checkpoint(
            args.rel_class_model
        )

        # Now that we have defined the argumentative components, we need to
        # build the argumentative structure. For this, we need first to build
        # the permutations of 2 components of all the relevant components
        source_components = []
        target_components = []
        for src, tgt in permutations(arg_comp, 2):
            source_components.append(src)
            target_components.append(tgt)

        logger.info("Classifying relations.")
        rel_classes = relation_classification(
            text=[arg_comp_annotations[comp['ref']]['text'] for comp in source_components],
            text_pair=[arg_comp_annotations[comp['ref']]['text'] for comp in target_components],
            model=rel_class_model,
            tokenizer=rel_class_model.config.name_or_path,
            id2label=rel_class_model.config.id2label,
            max_seq_length=args.max_seq_length,
            truncation_strategy=args.truncation_strategy,
            return_confidence=args.confidence,
            confidence_as_probability=args.confidence_as_probability
        )

        argumentative_structure = []
        for source, target, relation in zip(source_components, target_components, rel_classes):
            if args.filter_no_rel_class and relation == args.no_rel_class:
                continue
            annotation = {
                'source': source['id'],
                'target': target['id'],
                'relation': relation['label']
            }
            if args.confidence:
                annotation['confidence'] = relation['confidence']
            argumentative_structure.append(annotation)
        annotations['argumentative_structure'] = argumentative_structure

    logger.info(f"Saving annotated file to: {args.output_file}")
    with open(args.output_file, "wt") as fh:
        annotations = json.dump(annotations, fh)
