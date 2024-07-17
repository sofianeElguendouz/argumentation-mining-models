"""
Models module. It has the Transformers Modules of the Argumentation Mining Transformers library.
"""

from .relation_classification import RelationClassificationTransformerModule
from .sequence_tagging import SequenceTaggingTransformerModule
from .statement_classification import StatementClassificationTransformerModule

__all__ = [
    "RelationClassificationTransformerModule",
    "SequenceTaggingTransformerModule",
    "StatementClassificationTransformerModule",
]
