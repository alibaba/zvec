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

from zvec._zvec import _Doc

from .doc import Doc
from .schema import CollectionSchema


def convert_to_cpp_doc(doc: Doc, collection_schema: CollectionSchema) -> _Doc:
    if not doc or not collection_schema:
        return None

    _doc = _Doc()

    # set pk
    _doc.set_pk(doc.id)

    # set scalar fields
    for k, v in doc.fields.items():
        field_schema = collection_schema.field(k)
        if not field_schema:
            raise ValueError(
                f"schema validate failed: {k} not found in collection schema"
            )
        _doc.set_any(k, field_schema._get_object(), v)

    # set vector fields
    for k, v in doc.vectors.items():
        vector_schema = collection_schema.vector(k)
        if not vector_schema:
            raise ValueError(
                f"schema validate failed: {k} not found in collection schema"
            )
        _doc.set_any(k, vector_schema._get_object(), v)
    return _doc


def convert_to_py_doc(doc: _Doc, collection_schema: CollectionSchema) -> Doc:
    if not doc or not collection_schema:
        return None

    data_tuple = doc.get_all(collection_schema._get_object())
    return Doc._from_tuple(data_tuple)
