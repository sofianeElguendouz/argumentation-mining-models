"""
Models module. It has the Transformers Modules of the ACTA library.
"""

from .relation_classification import RelationClassificationTransformerModule
from .sequence_tagging import SequenceTaggingTransformerModule

__all__ = [
    "RelationClassificationTransformerModule",
    "SequenceTaggingTransformerModule"
]
