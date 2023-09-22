"""
Data utilities module. It has the Datasets and LightningDataModules of the ACTA library.
"""

from .relation_classification import RelationClassificationDataset
from .sequence_tagging import SequenceTaggingDataset

__all__ = [
    "RelationClassificationDataset",
    "SequenceTaggingDataset"
]
