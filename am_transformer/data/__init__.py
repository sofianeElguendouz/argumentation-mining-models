"""
Data utilities module. It has the Datasets and LightningDataModules of the ACTA library.
"""

from .relation_classification import RelationClassificationDataModule
from .sequence_tagging import SequenceTaggingDataModule

__all__ = [
    "RelationClassificationDataModule",
    "SequenceTaggingDataModule"
]
