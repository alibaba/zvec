# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Optional, TypeVar, Union

import numpy as np

# VectorType: DenseVectorType | SparseVectorType
DenseVectorType = Union[list[float], list[int], np.ndarray]
SparseVectorType = dict[int, float]
VectorType = Optional[Union[DenseVectorType, SparseVectorType]]

# Embeddable: Text | Image | Audio
TEXT = str
IMAGE = Union[str, bytes, np.ndarray]  # file path, raw bytes, or numpy array
AUDIO = Union[str, bytes, np.ndarray]  # file path, raw bytes, or numpy array

Embeddable = Optional[Union[TEXT, IMAGE, AUDIO]]

# Multimodal Embeddable
MD = TypeVar("MD", bound=Embeddable, contravariant=True)
