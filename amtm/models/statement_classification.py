"""
Pytorch Lightning Module for Statement Classification.

    Argumentation Mining Transformers Statement Classification Transformer Module
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

from .relation_classification import RelationClassificationTransformerModule


class StatementClassificationTransformerModule(RelationClassificationTransformerModule):
    """
    Lightning Module for classification of argumentative statements (e.g.,
    position, supporting argument, attacking argument, etc.).

    It's essentially a wrapper over the RelationClassificationTransformerModule
    since most of the work is done at dataset level. Please refer to the parent
    module for more information.
    """
